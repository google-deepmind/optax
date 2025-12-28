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
"""GaLore: Gradient Low-Rank Projection Optimizer.

Implementation of "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
Projection" (https://arxiv.org/abs/2403.03507) by Zhao et al.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


class GaLoreState(NamedTuple):
  """State for the GaLore optimizer.

  Attributes:
    count: Number of update steps taken.
    m: First moment estimate. For 2D params, stored in low-rank form.
    v: Second moment estimate. For 2D params, stored in low-rank form.
    projector: Projection matrices for each 2D parameter.

  """

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32
  m: base.Updates
  v: base.Updates
  projector: base.Updates  # Projection matrices P



def _get_orthogonal_matrix_left(
    weights: jax.Array,
    rank: int,
) -> jax.Array:
  """Compute left projection matrix using SVD (P from U).

  Args:
    weights: The gradient or weight matrix (2D), shape (m, n).
    rank: Target rank for projection.

  Returns:
    Orthogonal projection matrix P of shape (m, rank).
  """
  # SVD requires float32 (LAPACK doesn't support bfloat16), so cast and cast back
  original_dtype = weights.dtype
  weights_f32 = weights.astype(jnp.float32)
  u, _, _ = jnp.linalg.svd(weights_f32, full_matrices=False)
  return u[:, :rank].astype(original_dtype)


def _get_orthogonal_matrix_right(
    weights: jax.Array,
    rank: int,
) -> jax.Array:
  """Compute right projection matrix using SVD (P from V).

  Args:
    weights: The gradient or weight matrix (2D), shape (m, n).
    rank: Target rank for projection.

  Returns:
    Orthogonal projection matrix P of shape (n, rank).
  """
  # SVD requires float32 (LAPACK doesn't support bfloat16), so cast and cast back
  original_dtype = weights.dtype
  weights_f32 = weights.astype(jnp.float32)
  _, _, vh = jnp.linalg.svd(weights_f32, full_matrices=False)
  return vh[:rank, :].T.astype(original_dtype)


def _project_gradient_left(
    grad: jax.Array,
    projector: jax.Array,
) -> jax.Array:
  """Project gradient to low-rank subspace using left projection.

  Args:
    grad: Full gradient matrix (m x n).
    projector: Projection matrix P (m x r).

  Returns:
    Projected gradient (r x n).
  """
  return projector.T @ grad


def _project_gradient_right(
    grad: jax.Array,
    projector: jax.Array,
) -> jax.Array:
  """Project gradient to low-rank subspace using right projection.

  Args:
    grad: Full gradient matrix (m x n).
    projector: Projection matrix P (n x r).

  Returns:
    Projected gradient (m x r).
  """
  return grad @ projector


def _project_back_left(
    low_rank_update: jax.Array,
    projector: jax.Array,
) -> jax.Array:
  """Project low-rank update back to full space using left projection.

  Args:
    low_rank_update: Update in low-rank subspace (r x n).
    projector: Projection matrix P (m x r).

  Returns:
    Full-rank update (m x n).
  """
  return projector @ low_rank_update


def _project_back_right(
    low_rank_update: jax.Array,
    projector: jax.Array,
) -> jax.Array:
  """Project low-rank update back to full space using right projection.

  Args:
    low_rank_update: Update in low-rank subspace (m x r).
    projector: Projection matrix P (n x r).

  Returns:
    Full-rank update (m x n).
  """
  return low_rank_update @ projector.T


def scale_by_galore(
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformation:
  """Scale updates using GaLore (Gradient Low-Rank Projection).

  GaLore projects gradients of 2D weight matrices into a low-rank subspace,
  significantly reducing memory for optimizer states while maintaining
  full-parameter learning.

  For parameters of shape other than 2D (biases, layer norms), standard Adam updates are used
  without projection.

  Args:
    rank: Target rank for the low-rank projection. Lower rank = less memory
      but potentially slower convergence.
    update_proj_gap: Number of steps between projection matrix updates.
      The projection matrices are recomputed from gradient SVD every this
      many steps.
    scale: Scaling factor applied to the final updates.
    b1: Exponential decay rate for first moment estimate.
    b2: Exponential decay rate for second moment estimate.
    eps: Small constant for numerical stability.
    mu_dtype: Optional dtype for moment accumulators.

  Returns:
    A GradientTransformation implementing GaLore.

  References:
    Zhao et al., `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
    Projection <https://arxiv.org/abs/2403.03507>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params: base.Params) -> GaLoreState:
    # Use flattening to avoid brittle tuple-unpacking logic
    if not isinstance(rank, int):
      raise TypeError(f"`rank` must be an int, got {type(rank)}")
    if rank <= 0:
      raise ValueError(f"`rank` must be positive, got {rank}")

    leaves, treedef = jax.tree.flatten(params)

    m_list = []
    v_list = []
    proj_list = []

    for p in leaves:
      if p.ndim == 2:
        m_dim, n_dim = p.shape
        # Use left projection when m >= n, right when m < n
        # This minimizes moment memory: left gives (r, n), right gives (m, r)
        use_left = m_dim >= n_dim
        effective_rank = min(rank, m_dim, n_dim)

        if use_left:
          projector_shape = (m_dim, effective_rank)
          moment_shape = (effective_rank, n_dim)
        else:
          projector_shape = (n_dim, effective_rank)
          moment_shape = (m_dim, effective_rank)

        projector = jnp.zeros(projector_shape, dtype=p.dtype)
        m = jnp.zeros(moment_shape, dtype=mu_dtype or p.dtype)
        v = jnp.zeros(moment_shape, dtype=mu_dtype or p.dtype)
      else:
        # Non-2D parameters: use full-rank logic with dummy projector
        projector = jnp.zeros((0,0), dtype=p.dtype)
        m = jnp.zeros_like(p, dtype=mu_dtype or p.dtype)
        v = jnp.zeros_like(p, dtype=mu_dtype or p.dtype)

      m_list.append(m)
      v_list.append(v)
      proj_list.append(projector)

    return GaLoreState(
        count=jnp.zeros([], jnp.int32),
        m=treedef.unflatten(m_list),
        v=treedef.unflatten(v_list),
        projector=treedef.unflatten(proj_list),
    )

  def update_fn(
      updates: base.Updates,
      state: GaLoreState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, GaLoreState]:
    """Apply GaLore update."""
    del params
    count = state.count
    count_inc = numerics.safe_int32_increment(count)

    # Check if we should update projection matrices
    should_update_proj = (count % update_proj_gap) == 0

    def update_2d_left(grad, m, v, projector):
      """Update for 2D parameter with left projection."""
      # Compute effective rank from gradient dimensions (not projector shape
      # since projector could be (0,0) for non-2D params during tracing)
      effective_rank = min(rank, grad.shape[0], grad.shape[1])
      original_dtype = grad.dtype

      # Update projector if needed
      new_projector = jax.lax.cond(
          should_update_proj,
          lambda: _get_orthogonal_matrix_left(grad, effective_rank),
          lambda: projector,
      )

      # Project gradient to low-rank subspace
      low_rank_grad = _project_gradient_left(grad, new_projector)

      # Adam update in low-rank space
      new_m = b1 * m + (1 - b1) * low_rank_grad
      new_v = b2 * v + (1 - b2) * jnp.square(low_rank_grad)

      # Bias correction - compute in float32 then cast back for dtype stability
      bias_correction_m = (1 - b1 ** count_inc).astype(new_m.dtype)
      bias_correction_v = (1 - b2 ** count_inc).astype(new_v.dtype)
      m_hat = new_m / bias_correction_m
      v_hat = new_v / bias_correction_v

      # Normalized update in low-rank space
      low_rank_update = m_hat / (jnp.sqrt(v_hat) + eps)

      # Project back to full space and cast to original dtype
      full_update = (scale * _project_back_left(low_rank_update, new_projector)).astype(original_dtype)

      return full_update, new_m, new_v, new_projector

    def update_2d_right(grad, m, v, projector):
      """Update for 2D parameter with right projection."""
      # Compute effective rank from gradient dimensions (not projector shape
      # since projector could be (0,0) for non-2D params during tracing)
      effective_rank = min(rank, grad.shape[0], grad.shape[1])
      original_dtype = grad.dtype

      # Update projector if needed
      new_projector = jax.lax.cond(
          should_update_proj,
          lambda: _get_orthogonal_matrix_right(grad, effective_rank),
          lambda: projector,
      )

      # Project gradient to low-rank subspace
      low_rank_grad = _project_gradient_right(grad, new_projector)

      # Adam update in low-rank space
      new_m = b1 * m + (1 - b1) * low_rank_grad
      new_v = b2 * v + (1 - b2) * jnp.square(low_rank_grad)

      # Bias correction - compute in float32 then cast back for dtype stability
      bias_correction_m = (1 - b1 ** count_inc).astype(new_m.dtype)
      bias_correction_v = (1 - b2 ** count_inc).astype(new_v.dtype)
      m_hat = new_m / bias_correction_m
      v_hat = new_v / bias_correction_v

      # Normalized update in low-rank space
      low_rank_update = m_hat / (jnp.sqrt(v_hat) + eps)

      # Project back to full space and cast to original dtype
      full_update = (scale * _project_back_right(low_rank_update, new_projector)).astype(original_dtype)

      return full_update, new_m, new_v, new_projector

    def update_non2d(grad, m, v, projector):
      """Update for parameters except 2D (standard Adam)."""
      original_dtype = grad.dtype
      new_m = b1 * m + (1 - b1) * grad
      new_v = b2 * v + (1 - b2) * jnp.square(grad)

      # Bias correction - compute in float32 then cast back for dtype stability
      bias_correction_m = (1 - b1 ** count_inc).astype(new_m.dtype)
      bias_correction_v = (1 - b2 ** count_inc).astype(new_v.dtype)
      m_hat = new_m / bias_correction_m
      v_hat = new_v / bias_correction_v

      # Standard Adam update and cast to original dtype
      full_update = (scale * m_hat / (jnp.sqrt(v_hat) + eps)).astype(original_dtype)

      return full_update, new_m, new_v, projector

    def update_single_param(grad, m, v, projector):
      """Update a single parameter based on its type."""


      # We can't use jax.lax.cond with different return types, so we use
      # a switch that statically dispatches based on parameter structure.
      if grad.ndim != 2:
        return update_non2d(grad, m, v, projector)
      else:

        # This is determined at init time based on dimensions
        # Use left when m >= n (moments are r×n), right when m < n (moments are m×r)
        m_dim, n_dim = grad.shape
        if m_dim >= n_dim:
          return update_2d_left(grad, m, v, projector)
        else:
          return update_2d_right(grad, m, v, projector)

    # Map over all parameters
    results = jax.tree.map(
        update_single_param,
        updates,
        state.m,
        state.v,
        state.projector,

    )

    # Transpose results
    new_updates, new_m, new_v, new_projector = jax.tree.transpose(
        jax.tree.structure(updates),
        jax.tree.structure((0, 0, 0, 0)),
        results,
    )
    new_m = optax.tree.cast(new_m, mu_dtype)
    new_v = optax.tree.cast(new_v, mu_dtype)

    new_state = GaLoreState(
        count=count_inc,
        m=new_m,
        v=new_v,
        projector=new_projector,

    )

    return new_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def galore(
    learning_rate: base.ScalarOrSchedule,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: Optional[Any] = None,
    weight_decay: jax.typing.ArrayLike = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  r"""GaLore optimizer: Memory-efficient training via gradient low-rank projection.

  GaLore (Gradient Low-Rank Projection) is a memory-efficient training strategy
  that enables full-parameter learning while reducing optimizer state memory by
  projecting gradients into a low-rank subspace.

  The key insight is that gradients of weight matrices in neural networks often
  exhibit low-rank structure. GaLore exploits this by:

  1. Computing a low-rank projection matrix P using SVD of the gradient
  2. Projecting gradients to a low-rank subspace: R = P^T @ G (or G @ P)
  3. Maintaining optimizer states (m, v) in the reduced subspace
  4. Projecting updates back to full space: update = P @ normalized_R

  For a weight matrix of shape (m, n) with rank r projection:
  - Standard Adam stores m + v states: 2 * m * n parameters
  - GaLore stores: 2 * min(r*n, m*r) + projection matrix

  This can achieve up to 65% memory reduction for large linear layers.

  .. note::
    GaLore only projects 2D weight matrices. 1D parameters (biases, layer
    norms) use standard Adam updates without projection.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(jnp.square(x['w']))
    >>> solver = optax.contrib.galore(learning_rate=0.01, rank=16)
    >>> params = {'w': jnp.ones((100, 100)), 'b': jnp.ones((100,))}
    >>> print('Objective function: ', f(params))
    Objective function:  10000.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 9.80E+03
    Objective function: 9.60E+03
    Objective function: 9.41E+03
    Objective function: 9.22E+03
    Objective function: 9.04E+03

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    rank: Target rank for the low-rank projection. Lower values save more
      memory but may slow convergence. Default 128 is a good starting point.
    update_proj_gap: Number of steps between projection matrix updates.
      The projectors are recomputed from the gradient SVD every this many
      steps to adapt to the changing gradient landscape.
    scale: Additional scaling factor for updates.
    b1: Exponential decay rate for first moment (like Adam beta1).
    b2: Exponential decay rate for second moment (like Adam beta2).
    eps: Small constant for numerical stability in division.
    mu_dtype: Optional dtype for moment accumulators.
    weight_decay: Strength of weight decay regularization (decoupled, as in
      AdamW).
    mask: A tree with same structure as params PyTree, or a Callable that
      returns such a pytree. Leaves should be booleans indicating whether
      to apply weight decay to each parameter.

  Returns:
    A GradientTransformation implementing the GaLore optimizer.

  References:
    Zhao et al., `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
    Projection <https://arxiv.org/abs/2403.03507>`_, 2024
  """
  return combine.chain(
      scale_by_galore(
          rank=rank,
          update_proj_gap=update_proj_gap,
          scale=scale,
          b1=b1,
          b2=b2,
          eps=eps,
          mu_dtype=mu_dtype,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )
