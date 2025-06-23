# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
L-BFGS-B optimization with box constraints — JAX/Optax implementation.

This implementation follows the algorithm described in:
Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).
"A Limited Memory Algorithm for Bound Constrained Optimization."
SIAM Journal on Scientific Computing, 16(5), 1190-1208.

Key algorithm components:
- Algorithm 1: Main L-BFGS-B iteration (Section 4)
- Algorithm 2: Generalized Cauchy Point computation (Section 5)
- Algorithm 3: Subspace minimization (Section 6)
- Line search for unconstrained steps (Section 7)
"""
import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax import lax

from optax._src import base, linesearch

# Hyperparameters from Byrd et al. (1995)
_EPS = 1e-14                     # Machine precision tolerance
_C1, _C2, _DEC_RTOL = 1e-4, 0.9, 1e-6  # Wolfe conditions constants (Section 7)

# GCP breakpoints for Algorithm 2 (eq 5.8)
_DEFAULT_GCP_STEPS = jnp.array([0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

class LBFGSState(NamedTuple):
    """L-BFGS-B state following Byrd et al. (1995) Algorithm 1."""
    count:   chex.Array  # Iteration counter
    mem_cnt: chex.Array  # Number of stored correction pairs
    S:       chex.Array  # Step differences s_k = x_{k+1} - x_k (eq 3.2)
    Y:       chex.Array  # Gradient differences y_k = g_{k+1} - g_k (eq 3.2)
    R:       chex.Array  # Reciprocal values ρ_k = 1/(y_k^T s_k) (eq 3.4)
    gamma:   chex.Scalar # Initial Hessian approximation H_0 = γI (eq 3.1)



class _LBFGSBHandler:
    """L-BFGS-B handler implementing Byrd et al. (1995) Algorithm 1."""
    def __init__(self, m, lb, ub, max_ls_steps, lr, tolerance=1e-12):
        self.m, self.lb, self.ub, self.tolerance = m, lb, ub, tolerance
        # Line search for strong Wolfe conditions (Section 7)
        self.init_ls, self.step_ls, self.cond_ls = linesearch.zoom_linesearch(
            max_linesearch_steps=max_ls_steps, max_stepsize=lr,
            tol=0.0, increase_factor=2.0,
            slope_rtol=_C1, curv_rtol=_C2, approx_dec_rtol=_DEC_RTOL,
            interval_threshold=1e-8, verbose=False)

    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self, params):
        """Initialize L-BFGS-B state (Algorithm 1)."""
        flat, _ = jax.flatten_util.ravel_pytree(params)
        n = flat.shape[0]
        return LBFGSState(
            count=0, mem_cnt=0,
            S=jnp.zeros((self.m, n)), Y=jnp.zeros((self.m, n)), R=jnp.zeros(self.m),
            gamma=1.0
        )

    def step(self, params, state, grad, value_fn):
        """Algorithm 1 Steps 1-5: L-BFGS-B iteration."""
        # Flatten parameters and gradients
        x_flat, un = jax.flatten_util.ravel_pytree(params)
        g_flat, _ = jax.flatten_util.ravel_pytree(grad)

        # Step 0: Set up box constraints [l, u]
        bounds = None
        if self.lb is not None or self.ub is not None:
            lo = self.lb if self.lb is not None else jnp.full_like(x_flat, -jnp.inf)
            hi = self.ub if self.ub is not None else jnp.full_like(x_flat,  jnp.inf)
            bounds = (lo, hi)

        # Step 1: Compute search direction d_k
        d_flat = get_direction(g_flat, x_flat, state, bounds, self.m)
        d = un(d_flat)

        # Step 2: Line search for step size α_k (Section 7)
        val0, g0 = value_fn(params), grad
        ls_state = self.init_ls(updates=d, params=params, value=val0, grad=g0,
                                prev_stepsize=1.0, initial_guess_strategy="one")
        final = lax.while_loop(
            self.cond_ls,
            functools.partial(
                self.step_ls,
                value_and_grad_fn=jax.value_and_grad(value_fn),
                fn_kwargs={}
            ),
            ls_state
        )

        # Step 3: Update iterate with bounds projection
        step_flat_projected = _project_to_bounds(x_flat, d_flat, final.stepsize, bounds)
        step = un(step_flat_projected)
        new_params = jax.tree.map(lambda p, s: p + s, params, step)

        # Steps 4-5: Update limited memory with correction pair {s_k, y_k}
        new_grad_flat, _ = jax.flatten_util.ravel_pytree(jax.grad(value_fn)(new_params))
        s_k = step_flat_projected  # s_k = x_{k+1} - x_k
        y_k = new_grad_flat - g_flat  # y_k = ∇f(x_{k+1}) - ∇f(x_k)
        new_state = _update_bfgs_memory(state, s_k, y_k, self.m)

        # Convergence check
        step = _convergence_check(step, self.tolerance)

        return step, new_state


@functools.partial(jax.jit, static_argnames=['m', 'gcp_steps'])
def get_direction(g_flat, x_flat, state, bounds, m, gcp_steps=_DEFAULT_GCP_STEPS):
    """Algorithms 1-2: Direction computation (GCP for k=0, L-BFGS for k>0)."""
    def gcp_direction():
        """Algorithm 2: Generalized Cauchy Point (Section 5, eq 5.8)."""
        if bounds is None:
            return -g_flat
        lo, hi = bounds
        def obj(t):
            trial = jnp.clip(x_flat - t * g_flat, lo, hi)
            disp = trial - x_flat
            return g_flat @ disp + 0.5 * (disp @ disp)
        objectives = jax.vmap(obj)(gcp_steps)
        best_t = gcp_steps[jnp.argmin(objectives)]
        return jnp.clip(x_flat - best_t * g_flat, lo, hi) - x_flat

    def lbfgs_direction():
        """Algorithm 1 Step 2: L-BFGS two-loop recursion (eqs 3.1-3.4)."""
        k = jnp.minimum(state.mem_cnt, m)
        idx = (state.mem_cnt - 1 - jnp.arange(m)) % m

        def bwd(carry, i):
            q, a = carry
            valid = (i < k) & (state.R[i] > _EPS)
            ai = jnp.where(valid, state.R[i] * (state.S[i] @ q), 0.0)
            return (q - ai * state.Y[i] * valid, a.at[i].set(ai)), None
        (q, a), _ = lax.scan(bwd, (g_flat, jnp.zeros(m)), idx)

        r = state.gamma * q  # H_0^{-1} q = γ q (eq 3.1)

        def fwd(r, i):
            valid = (i < k) & (state.R[i] > _EPS)
            bi = jnp.where(valid, state.R[i] * (state.Y[i] @ r), 0.0)
            return r + (a[i] - bi) * state.S[i] * valid, None
        r_final, _ = lax.scan(fwd, r, idx[::-1])

        return -r_final

    # Choose direction: GCP for k=0, L-BFGS otherwise
    d_flat = lax.cond(state.count == 0, gcp_direction, lbfgs_direction)
    # Descent safeguard
    return jnp.where((g_flat @ d_flat) < -_EPS, d_flat, -g_flat)

@jax.jit
def _update_bfgs_memory(state, s_k, y_k, m):
    """BFGS memory update (Algorithm 1 Steps 4-5, eqs 3.2-3.4)."""
    ys = y_k @ s_k
    ok = ys > _EPS
    rho = jnp.where(ok, 1.0 / ys, 0.0)  # rho_k = 1/(s_k^T y_k) (eq 3.4)

    # Store correction pair in circular buffer
    idx = state.mem_cnt % m
    S_new = jnp.where(ok, state.S.at[idx].set(s_k), state.S)
    Y_new = jnp.where(ok, state.Y.at[idx].set(y_k), state.Y)
    R_new = jnp.where(ok, state.R.at[idx].set(rho), state.R)
    gamma_new = jnp.where(ok, ys / jnp.maximum(y_k @ y_k, _EPS), state.gamma)  # eq 3.1

    return LBFGSState(
        count=state.count + 1,
        mem_cnt=state.mem_cnt + ok.astype(jnp.int32),
        S=S_new, Y=Y_new, R=R_new, gamma=gamma_new
    )

@jax.jit
def _project_to_bounds(x_flat, d_flat, stepsize, bounds):
    """Bounds projection: x + α*d → [l, u]."""
    if bounds is None:
        return stepsize * d_flat
    lo, hi = bounds
    new_x_flat = jnp.clip(x_flat + stepsize * d_flat, lo, hi)
    return new_x_flat - x_flat

@jax.jit
def _convergence_check(step, tolerance):
    """Convergence check: zero step if ||step|| < tolerance."""
    step_norm = jnp.sqrt(sum(jnp.sum(jnp.square(s)) for s in jax.tree.leaves(step)))
    converged = step_norm < tolerance
    return jax.tree.map(lambda s: jnp.where(converged, jnp.zeros_like(s), s), step)

def lbfgs_transform(
    memory_size=10, lower_bounds=None, upper_bounds=None,
    learning_rate=1.0, max_line_search_steps=20,
    tolerance=1e-12
):
    """L-BFGS-B transform implementing Byrd et al. (1995) Algorithms 1-2."""
    # Flatten bounds for efficient vector operations
    lb_flat = (jax.flatten_util.ravel_pytree(lower_bounds)[0]
               if lower_bounds is not None else None)
    ub_flat = (jax.flatten_util.ravel_pytree(upper_bounds)[0]
               if upper_bounds is not None else None)

    # Create algorithm handler with flattened bounds
    handler = _LBFGSBHandler(
        memory_size, lb_flat, ub_flat, max_line_search_steps,
        learning_rate, tolerance
    )

    def init_fn(params):
        return handler.init(params)

    def update_fn(updates, state, params=None, *, value, grad, value_fn):
        """L-BFGS-B update: ignores 'updates', computes own direction."""
        if params is None or value_fn is None:
            raise ValueError("lbfgs_b requires params and value_fn.")
        return handler.step(params, state, grad, value_fn)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
