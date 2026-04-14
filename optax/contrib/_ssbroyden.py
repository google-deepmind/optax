# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Self-Scaled Broyden (SSBroyden/SSBFGS) optimizer.

Ported from the PyTorch implementation provided by
SciMBA (https://www.scimba.org/).

Based on:
  Urbán, J. F., Stefanou, P., & Pons, J. A. (2025).
  Unveiling the optimization process of physics informed neural networks:
  How accurate and competitive can PINNs be?.
  Journal of Computational Physics, 523, 113656.
"""

from typing import NamedTuple, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import linesearch as _linesearch
from optax._src import numerics
from optax._src import transform
import optax.tree


class ScaleBySSQuasiNewtonState(NamedTuple):
    """State for the Self-Scaled Broyden/BFGS solver.

    Attributes:
      count: iteration counter.
      flat_params: flattened parameters from the previous step.
      flat_updates: flattened gradients from the previous step.
      hessian_inv: dense inverse Hessian approximation, shape ``[n, n]``.
      flat_prec_grad: flattened preconditioned gradient ``H_k g_k`` from the
        previous step, used to recover the effective step size.
    """

    count: jax.typing.ArrayLike
    flat_params: jax.typing.ArrayLike
    flat_updates: jax.typing.ArrayLike
    hessian_inv: jax.typing.ArrayLike
    flat_prec_grad: jax.typing.ArrayLike


def _update_hessian_inv(
    hessian_inv: jax.Array,
    s_k: jax.Array,
    y_k: jax.Array,
    alpha_k: jax.Array,
    flat_prev_grad: jax.Array,
    method: str,
) -> jax.Array:
    """Update the inverse Hessian approximation using SS-BFGS or SS-Broyden."""
    n = s_k.shape[0]
    Hk_yk = hessian_inv @ y_k
    yk_dot_Hkyk = y_k @ Hk_yk
    yk_dot_sk = y_k @ s_k

    # Safeguard denominators
    safe_yk_dot_Hkyk = jnp.where(
        jnp.abs(yk_dot_Hkyk) > 0, yk_dot_Hkyk, jnp.ones_like(yk_dot_Hkyk)
    )
    safe_yk_dot_sk = jnp.where(
        jnp.abs(yk_dot_sk) > 0, yk_dot_sk, jnp.ones_like(yk_dot_sk)
    )

    v_k = jnp.sqrt(jnp.maximum(yk_dot_Hkyk, 0.0)) * (
        s_k / safe_yk_dot_sk - Hk_yk / safe_yk_dot_Hkyk
    )

    # SS-BFGS tau
    sk_dot_gk = s_k @ flat_prev_grad
    denominator = alpha_k * sk_dot_gk
    safe_denom = jnp.where(jnp.abs(denominator) > 0, denominator, 1.0)
    tau_k = jnp.where(
        jnp.abs(denominator) > 0,
        jnp.minimum(1.0, -yk_dot_sk / safe_denom),
        1.0,
    )
    phi_k = jnp.ones_like(tau_k)

    if method == "ssbroyden":
        b_k = -alpha_k * sk_dot_gk / safe_yk_dot_sk
        h_k = yk_dot_Hkyk / safe_yk_dot_sk
        a_k = h_k * b_k - 1.0
        # c_k = sqrt(a_k / (a_k + 1));
        # guard against negative or zero denominator
        safe_a_k_ratio = jnp.where(
            (a_k > 0) & (a_k + 1.0 > 0),
            a_k / (a_k + 1.0),
            jnp.zeros_like(a_k),
        )
        c_k = jnp.sqrt(jnp.maximum(safe_a_k_ratio, 0.0))
        rhom_k = jnp.minimum(1.0, h_k * (1.0 - c_k))

        safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
        thetam_k = (rhom_k - 1.0) / safe_a_k
        safe_rhom_k = jnp.where(
            jnp.abs(rhom_k) > 0, rhom_k, jnp.ones_like(rhom_k)
        )
        thetap_k = 1.0 / safe_rhom_k

        safe_b_k = jnp.where(
            jnp.abs(b_k) > 0, b_k, jnp.ones_like(b_k)
        )
        theta_k = jnp.maximum(
            thetam_k,
            jnp.minimum(thetap_k, (1.0 - b_k) / safe_b_k),
        )

        sigma_k = 1.0 + a_k * theta_k
        exp = -1.0 / jnp.maximum(n - 1, 1)
        sigma_k_pow = jnp.where(
            n > 1, sigma_k ** exp, 1.0
        )

        safe_theta_k = jnp.where(
            jnp.abs(theta_k) > 0,
            theta_k,
            jnp.ones_like(theta_k),
        )
        tau_k = jnp.where(
            theta_k > 0,
            tau_k * jnp.minimum(sigma_k_pow, 1.0 / safe_theta_k),
            jnp.minimum(tau_k * sigma_k_pow, sigma_k),
        )
        phi_k = (1.0 - theta_k) / (1.0 + a_k * theta_k)

    safe_tau_k = jnp.where(jnp.abs(tau_k) > 0, tau_k, jnp.ones_like(tau_k))

    temp1 = jnp.outer(Hk_yk, Hk_yk) / safe_yk_dot_Hkyk
    temp2 = phi_k * jnp.outer(v_k, v_k)
    temp3 = jnp.outer(s_k, s_k) / safe_yk_dot_sk
    H_new = (1.0 / safe_tau_k) * (hessian_inv - temp1 + temp2) + temp3

    # Guard against NaN or non-positive curvature
    valid = (
        ~jnp.any(jnp.isnan(H_new))
        & ~jnp.any(jnp.isinf(H_new))
        & (yk_dot_sk > 0)
    )
    H_new = jnp.where(valid, H_new, hessian_inv)
    return H_new


def scale_by_ss_quasi_newton(
    method: str = "ssbfgs",
    scale_init_precond: bool = True,
) -> base.GradientTransformation:
    r"""Scale updates by the Self-Scaled Broyden/BFGS inverse Hessian.

    This maintains a dense approximation of the inverse Hessian
    :math:`H_k` and returns the preconditioned gradient
    :math:`H_k \nabla f(w_k)`. Unlike L-BFGS, the full
    :math:`n \times n` matrix is stored, so this is suitable only for
    small to medium scale problems.

    The inverse Hessian is updated using a self-scaled formula:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + \phi_k v_k v_k^\top) + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factors :math:`\tau_k` and :math:`\phi_k` depend on
    the chosen ``method`` (``'ssbfgs'`` or ``'ssbroyden'``).

    This function is typically chained with a line search such as
    :func:`optax.scale_by_zoom_linesearch`.

    Args:
      method: ``'ssbfgs'`` or ``'ssbroyden'``.
        Controls the self-scaling formula.
      scale_init_precond: if ``True``, scale the initial identity
        preconditioner
        by a capped reciprocal of the gradient norm at the first step.

    Returns:
      A :class:`optax.GradientTransformation` object.

    References:
      Urbán et al, `Unveiling the optimization process of physics
      informed neural networks: How accurate and competitive can
      PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium scale
      problems.
    """
    if method not in ("ssbfgs", "ssbroyden"):
        raise ValueError(
            f"method must be 'ssbfgs' or 'ssbroyden',"
            f" got '{method}'"
        )

    def init_fn(params: base.Params) -> ScaleBySSQuasiNewtonState:
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        n = flat_params.shape[0]
        dtype = flat_params.dtype
        return ScaleBySSQuasiNewtonState(
            count=jnp.zeros([], jnp.int32),
            flat_params=jnp.zeros(n, dtype=dtype),
            flat_updates=jnp.zeros(n, dtype=dtype),
            hessian_inv=jnp.eye(n, dtype=dtype),
            flat_prec_grad=jnp.zeros(n, dtype=dtype),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleBySSQuasiNewtonState,
        params: base.Params,
    ) -> tuple[base.Updates, ScaleBySSQuasiNewtonState]:
        flat_updates_new, unravel_fn = jax.flatten_util.ravel_pytree(updates)
        flat_params_new = jax.flatten_util.ravel_pytree(params)[0]

        # --- 1. Update inverse Hessian using info from previous step ---
        s_k = flat_params_new - state.flat_params
        y_k = flat_updates_new - state.flat_updates

        # Zero out at first step (no previous data)
        s_k = jnp.where(state.count > 0, s_k, jnp.zeros_like(s_k))
        y_k = jnp.where(state.count > 0, y_k, jnp.zeros_like(y_k))

        # Recover the effective step size from the previous
        # iteration.  If the chain is
        # scale_by_ss -> scale(-1) -> linesearch, then
        # s_k = -alpha * prec_grad, so
        # alpha = -(s_k . prec_grad) / ||prec_grad||^2
        prec_sq = state.flat_prec_grad @ state.flat_prec_grad
        alpha_k = jnp.where(
            prec_sq > 0,
            -(s_k @ state.flat_prec_grad) / prec_sq,
            0.0,
        )

        hessian_inv = jnp.where(
            state.count > 0,
            _update_hessian_inv(
                state.hessian_inv,
                s_k,
                y_k,
                alpha_k,
                state.flat_updates,
                method,
            ),
            state.hessian_inv,
        )

        # --- 2. Scale the initial preconditioner at the first step ---
        if scale_init_precond:
            update_norm = jnp.linalg.norm(flat_updates_new)
            capped_inv_norm = jnp.minimum(
                1.0, 1.0 / jnp.maximum(update_norm, 1e-30)
            )
            hessian_inv = jnp.where(
                state.count == 0, capped_inv_norm * hessian_inv, hessian_inv
            )

        # --- 3. Precondition the current gradient ---
        prec_grad_flat = hessian_inv @ flat_updates_new
        prec_grad = unravel_fn(prec_grad_flat)

        new_state = ScaleBySSQuasiNewtonState(
            count=numerics.safe_increment(state.count),
            flat_params=flat_params_new,
            flat_updates=flat_updates_new,
            hessian_inv=hessian_inv,
            flat_prec_grad=prec_grad_flat,
        )
        return prec_grad, new_state

    return base.GradientTransformation(init_fn, update_fn)


def ssbroyden(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    scale_init_precond: bool = True,
    linesearch: Optional[
        Union[base.GradientTransformationExtraArgs, base.GradientTransformation]
    ] = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=20, initial_guess_strategy="one"
    ),
) -> base.GradientTransformationExtraArgs:
    r"""Self-Scaled Broyden optimizer.

    SSBroyden is a quasi-Newton method that maintains a dense approximation of
    the inverse Hessian. Unlike L-BFGS which uses a limited-memory
    approximation, this method stores the full :math:`n \times n` matrix and
    updates it using the self-scaled Broyden formula, yielding improved scaling
    of the search direction at the cost of higher memory usage.

    The inverse Hessian :math:`H_k` is updated each iteration as:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}\!\left(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + \phi_k\, v_k v_k^\top\right)
      + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factors :math:`\tau_k, \phi_k` use the Broyden
    formula.

    Args:
      learning_rate: optional global scaling factor, either fixed or evolving
        along iterations with a scheduler. By default the learning rate is
        handled by the line search.
      scale_init_precond: whether to scale the initial identity
        preconditioner by a capped reciprocal of the gradient norm.
      linesearch: an instance of
        :class:`optax.GradientTransformationExtraArgs` such as
        :func:`optax.scale_by_zoom_linesearch` that computes a
        step size satisfying sufficient decrease and curvature
        conditions. Pass ``None`` to disable the line search.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs` object.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)
      >>> solver = optax.contrib.ssbroyden()
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> value_and_grad = optax.value_and_grad_from_state(f)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...      grad, opt_state, params,
      ...      value=value, grad=grad, value_fn=f,
      ...   )
      ...   params = optax.apply_updates(params, updates)

    References:
      Urbán et al, `Unveiling the optimization process of
      physics informed neural networks: How accurate and
      competitive can PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium
      scale problems.

    .. warning::
      This optimizer works best with a line search (default is a zoom line
      search). See the example above for best use in a non-stochastic setting
      where gradients computed by the line search can be recycled using
      :func:`optax.value_and_grad_from_state`.

    .. seealso:: :func:`optax.contrib.ssbfgs`
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = transform.scale_by_learning_rate(learning_rate)
    if linesearch is None:
        linesearch = base.identity()
    return combine.chain(
        scale_by_ss_quasi_newton(
            method="ssbroyden", scale_init_precond=scale_init_precond
        ),
        base_scaling,
        linesearch,
    )


def ssbfgs(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    scale_init_precond: bool = True,
    linesearch: Optional[
        Union[base.GradientTransformationExtraArgs, base.GradientTransformation]
    ] = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=20, initial_guess_strategy="one"
    ),
) -> base.GradientTransformationExtraArgs:
    r"""Self-Scaled BFGS optimizer.

    SSBFGS is a quasi-Newton method that maintains a dense approximation of
    the inverse Hessian. Unlike L-BFGS which uses a limited-memory
    approximation, this method stores the full :math:`n \times n` matrix and
    updates it using the self-scaled BFGS formula, yielding improved scaling
    of the search direction at the cost of higher memory usage.

    The inverse Hessian :math:`H_k` is updated each iteration as:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}\!\left(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + v_k v_k^\top\right)
      + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factor :math:`\tau_k` uses the BFGS formula
    (:math:`\phi_k = 1`).

    Args:
      learning_rate: optional global scaling factor, either fixed or evolving
        along iterations with a scheduler. By default the learning rate is
        handled by the line search.
      scale_init_precond: whether to scale the initial identity
        preconditioner by a capped reciprocal of the gradient norm.
      linesearch: an instance of
        :class:`optax.GradientTransformationExtraArgs` such as
        :func:`optax.scale_by_zoom_linesearch` that computes a
        step size satisfying sufficient decrease and curvature
        conditions. Pass ``None`` to disable the line search.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs` object.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)
      >>> solver = optax.contrib.ssbfgs()
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> value_and_grad = optax.value_and_grad_from_state(f)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...      grad, opt_state, params,
      ...      value=value, grad=grad, value_fn=f,
      ...   )
      ...   params = optax.apply_updates(params, updates)

    References:
      Urbán et al, `Unveiling the optimization process of
      physics informed neural networks: How accurate and
      competitive can PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium
      scale problems.

    .. warning::
      This optimizer works best with a line search (default is a zoom line
      search). See the example above for best use in a non-stochastic setting
      where gradients computed by the line search can be recycled using
      :func:`optax.value_and_grad_from_state`.

    .. seealso:: :func:`optax.contrib.ssbroyden`
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = transform.scale_by_learning_rate(learning_rate)
    if linesearch is None:
        linesearch = base.identity()
    return combine.chain(
        scale_by_ss_quasi_newton(
            method="ssbfgs", scale_init_precond=scale_init_precond
        ),
        base_scaling,
        linesearch,
    )
