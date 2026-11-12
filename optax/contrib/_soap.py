# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""SOAP: Improving and Stabilizing Shampoo using Adam.

Implementation of "SOAP: Improving and Stabilizing Shampoo using Adam"
(https://arxiv.org/abs/2409.11321) by Nikhil Vyas, Depen Morwani, Rosie Zhao,
Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _adding
import optax.tree


class ScaleBySOAPState(NamedTuple):
  """State for the SOAP optimizer."""

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32
  # Kronecker factors and their eigenbases, stored in float32.
  # Leaves have shape (m, m) / (n, n) for 2D params, or (0,) for others.
  left_factor: base.Updates
  right_factor: base.Updates
  left_basis: base.Updates
  right_basis: base.Updates
  # Adam moment buffers maintained in the rotated subspace for 2D params,
  # or in the original space for non-2D params.
  mu: base.Updates
  nu: base.Updates


def scale_by_soap(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    precondition_frequency: int = 10,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""Scale updates using the SOAP preconditioner.

  See :func:`optax.contrib.soap` for full details.

  Args:
    b1: Decay rate for the first moment (momentum) estimates.
    b2: Decay rate for the second moment and Kronecker factor estimates.
    eps: Small constant added to the denominator for numerical stability.
    precondition_frequency: Number of steps between eigenbasis recomputations.
      Lower values track gradient geometry more closely at higher cost per step.
      Must be a positive Python int (not a JAX-traced value).
    mu_dtype: Optional dtype for the first moment buffer. Useful for
      reducing memory in mixed-precision training. If ``None``, the dtype is
      inferred from the parameters.

  Returns:
    A :class:`optax.GradientTransformation`.
  """
  if not isinstance(precondition_frequency, int) or precondition_frequency < 1:
    raise ValueError(
        f'`precondition_frequency` must be a positive int, got'
        f' {precondition_frequency!r}.'
    )

  # Normalize to float32 so that Python-float and JAX-float32 closures compute
  # (1 - b2) identically. Without this, Python-float b2=0.999 gives
  # 1-b2=0.001 (Python arithmetic) while JAX float32(0.999) gives
  # 1-b2=0.0009999871 (float32 arithmetic), causing Kronecker factors to differ
  # and the null-space eigenvectors of R to be numerically arbitrary (any
  # orthonormal basis of a degenerate subspace is valid, so even tiny matrix
  # differences produce completely different eigenvectors).
  b1 = jnp.asarray(b1, dtype=jnp.float32)
  b2 = jnp.asarray(b2, dtype=jnp.float32)
  eps = jnp.asarray(eps, dtype=jnp.float32)

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params: base.Params) -> ScaleBySOAPState:
    param_leaves = jax.tree.leaves(params)
    if not param_leaves:
      # Called with placeholder params (e.g. from tree_map_params). Return an
      # empty state that matches the empty tree structure.
      empty = params
      return ScaleBySOAPState(
          count=jnp.zeros([], jnp.int32),
          left_factor=empty,
          right_factor=empty,
          left_basis=empty,
          right_basis=empty,
          mu=empty,
          nu=empty,
      )

    def _init_factors(p):
      if p.ndim == 2:
        m, n = p.shape
        return (
            jnp.zeros((m, m), dtype=jnp.float32),
            jnp.zeros((n, n), dtype=jnp.float32),
            jnp.eye(m, dtype=jnp.float32),
            jnp.eye(n, dtype=jnp.float32),
        )
      return (
          jnp.zeros((0,), dtype=jnp.float32),
          jnp.zeros((0,), dtype=jnp.float32),
          jnp.zeros((0,), dtype=jnp.float32),
          jnp.zeros((0,), dtype=jnp.float32),
      )

    factor_tuples = jax.tree.map(_init_factors, params)
    lf, rf, lb, rb = jax.tree.transpose(
        jax.tree.structure(params),
        jax.tree.structure((0, 0, 0, 0)),
        factor_tuples,
    )

    return ScaleBySOAPState(
        count=jnp.zeros([], jnp.int32),
        left_factor=lf,
        right_factor=rf,
        left_basis=lb,
        right_basis=rb,
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        nu=optax.tree.zeros_like(params),
    )

  def update_fn(
      updates: base.Updates,
      state: ScaleBySOAPState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, ScaleBySOAPState]:
    del params
    count_inc = numerics.safe_int32_increment(state.count)
    should_update_bases = jnp.equal(
        jnp.mod(state.count, precondition_frequency), 0
    )

    def _update_2d(g, l, r, q_l, q_r, mu, nu):
      g_f32 = g.astype(jnp.float32)

      l_new = b2 * l + (1.0 - b2) * (g_f32 @ g_f32.T)
      r_new = b2 * r + (1.0 - b2) * (g_f32.T @ g_f32)

      def _recompute_bases(args):
        l_cur, r_cur = args
        # Symmetrize to guard against accumulated floating-point asymmetry.
        _, new_q_l = jnp.linalg.eigh((l_cur + l_cur.T) * 0.5)
        _, new_q_r = jnp.linalg.eigh((r_cur + r_cur.T) * 0.5)
        return new_q_l, new_q_r

      q_l_new, q_r_new = jax.lax.cond(
          should_update_bases,
          _recompute_bases,
          lambda args: (q_l, q_r),
          (l_new, r_new),
      )

      g_proj = q_l_new.T @ g_f32 @ q_r_new

      mu_f32 = mu.astype(jnp.float32)
      nu_f32 = nu.astype(jnp.float32)

      mu_new = b1 * mu_f32 + (1.0 - b1) * g_proj
      nu_new = b2 * nu_f32 + (1.0 - b2) * jnp.square(g_proj)

      mu_hat = mu_new / (1.0 - b1**count_inc)
      nu_hat = nu_new / (1.0 - b2**count_inc)
      u_proj = mu_hat / (jnp.sqrt(nu_hat) + eps)

      u = (q_l_new @ u_proj @ q_r_new.T).astype(g.dtype)
      return (
          u,
          l_new,
          r_new,
          q_l_new,
          q_r_new,
          mu_new.astype(mu.dtype),
          nu_new.astype(nu.dtype),
      )

    def _update_nond(g, mu, nu):
      g_f32 = g.astype(jnp.float32)
      mu_new = b1 * mu.astype(jnp.float32) + (1.0 - b1) * g_f32
      nu_new = b2 * nu.astype(jnp.float32) + (1.0 - b2) * jnp.square(g_f32)
      mu_hat = mu_new / (1.0 - b1**count_inc)
      nu_hat = nu_new / (1.0 - b2**count_inc)
      u = (mu_hat / (jnp.sqrt(nu_hat) + eps)).astype(g.dtype)
      return u, mu_new.astype(mu.dtype), nu_new.astype(nu.dtype)

    def _update_single(g, l, r, q_l, q_r, mu, nu):
      if g.ndim == 2:
        u, l_new, r_new, q_l_new, q_r_new, mu_new, nu_new = _update_2d(
            g, l, r, q_l, q_r, mu, nu
        )
      else:
        u, mu_new, nu_new = _update_nond(g, mu, nu)
        l_new, r_new, q_l_new, q_r_new = l, r, q_l, q_r
      return u, l_new, r_new, q_l_new, q_r_new, mu_new, nu_new

    result_tuples = jax.tree.map(
        _update_single,
        updates,
        state.left_factor,
        state.right_factor,
        state.left_basis,
        state.right_basis,
        state.mu,
        state.nu,
    )

    new_updates, new_lf, new_rf, new_lb, new_rb, new_mu, new_nu = (
        jax.tree.transpose(
            jax.tree.structure(updates),
            jax.tree.structure((0, 0, 0, 0, 0, 0, 0)),
            result_tuples,
        )
    )

    new_mu = optax.tree.cast(new_mu, mu_dtype)

    return new_updates, ScaleBySOAPState(
        count=count_inc,
        left_factor=new_lf,
        right_factor=new_rf,
        left_basis=new_lb,
        right_basis=new_rb,
        mu=new_mu,
        nu=new_nu,
    )

  # pyrefly: ignore[bad-argument-type]
  return base.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    precondition_frequency: int = 10,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""SOAP: Improving and Stabilizing Shampoo using Adam.

  SOAP combines the full-matrix preconditioning of Shampoo with the adaptive
  moment estimation of Adam. For each 2D weight matrix :math:`W \in
  \mathbb{R}^{m \times n}`, it maintains Kronecker-factor matrices whose
  eigenbases define a rotation of the gradient. Adam's moment buffers are then
  maintained in this rotated space, so the effective preconditioner is
  Kronecker-structured but the per-coordinate adaptivity comes from Adam.

  For a 2D parameter with gradient :math:`G_t`:

  .. math::

    \begin{align*}
      L_t &\leftarrow \beta_2 L_{t-1} + (1 - \beta_2) G_t G_t^\top \\
      R_t &\leftarrow \beta_2 R_{t-1} + (1 - \beta_2) G_t^\top G_t \\
      Q_L, Q_R &\leftarrow \text{eigh}(L_t),\, \text{eigh}(R_t)
        \quad (\text{every } k \text{ steps}) \\
      \tilde{G}_t &\leftarrow Q_L^\top G_t Q_R \\
      m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \tilde{G}_t \\
      v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \tilde{G}_t^2 \\
      \hat{m}_t &\leftarrow m_t / (1 - \beta_1^t) \\
      \hat{v}_t &\leftarrow v_t / (1 - \beta_2^t) \\
      \Delta_t &\leftarrow Q_L \bigl(
        \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon) \bigr) Q_R^\top
    \end{align*}

  Parameters with fewer than 2 dimensions fall back to standard Adam.

  .. note::
    SOAP stores left and right Kronecker factors and their eigenbases for each
    2D parameter, introducing memory overhead of :math:`O(m^2 + n^2)` per
    parameter on top of the :math:`O(mn)` Adam moments. For very large weight
    matrices this can be substantial; consider using it selectively via
    :func:`optax.masked`.

  .. note::
    Eigenbasis recomputation via ``eigh`` adds per-step cost every
    ``precondition_frequency`` steps. The default of 10 balances tracking
    quality against compute. For large layers, increase this to 50–100.

  Args:
    learning_rate: A global scaling factor, either fixed or a schedule; see
      :func:`optax.scale_by_learning_rate`.
    b1: Decay rate for the first moment (momentum) estimates.
    b2: Decay rate for the second moment and Kronecker factor estimates.
    eps: Small constant added to the denominator for numerical stability.
    weight_decay: Optional :math:`\ell_2` regularization strength.
    weight_decay_mask: A tree with the same structure as (or a prefix of) the
      params pytree, or a callable that returns such a tree given the params.
      Leaves should be booleans indicating which parameters to apply weight
      decay to.
    precondition_frequency: Number of steps between eigenbasis recomputations
      from the Kronecker factors. Must be a positive Python int.
    mu_dtype: Optional dtype for the first moment buffer; useful for reducing
      memory in mixed-precision training.

  Returns:
    A :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def loss(params):
    ...   return jnp.sum(jnp.square(params['w'] - jnp.ones((4, 4))))
    >>> params = {'w': jnp.zeros((4, 4)), 'b': jnp.zeros(4)}
    >>> solver = optax.contrib.soap(learning_rate=1e-2)
    >>> state = solver.init(params)
    >>> for _ in range(5):
    ...   grads = jax.grad(loss)(params)
    ...   updates, state = solver.update(grads, state, params)
    ...   params = optax.apply_updates(params, updates)

  References:
    Vyas et al., `SOAP: Improving and Stabilizing Shampoo using Adam
    <https://arxiv.org/abs/2409.11321>`_, 2024
  """
  return combine.chain(
      scale_by_soap(
          b1=b1,
          b2=b2,
          eps=eps,
          precondition_frequency=precondition_frequency,
          mu_dtype=mu_dtype,
      ),
      # pyrefly: ignore[bad-argument-type]
      _adding.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate),
  )
