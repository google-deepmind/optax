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

import math
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform


ReshapeFn = Callable[[jax.Array], jax.Array]


class GaLoreDimensionNumbers(NamedTuple):
  """Specification for which weight axes form the 2D matrix for GaLore.

  For tensors that are logically 2D linear maps but stored as higher-dimensional
  arrays (e.g., attention projections stored as [embedding, heads, head_dim]),
  this specifies which axes should be treated as the "reduction" (input) and
  "output" dimensions for low-rank projection.

  The tensor is reshaped to (reduction, output) 2D form before SVD projection,
  then reshaped back to original form after projection.

  Example:
    For attention weights with shape (embed_dim, num_heads, head_dim):
    - reduction_axis=0 treats embed_dim as the input dimension
    - output_axis=(1, 2) treats heads*head_dim as the output dimension

  Attributes:
    reduction_axis: Axis or axes representing the input/reduction dimension.
    output_axis: Axis or axes representing the output dimension.
  """
  reduction_axis: Union[Sequence[int], int] = 0
  output_axis: Union[Sequence[int], int] = 1


# Type for dimension numbers: can be a spec, a pytree of specs, or a callable
GaLoreDimNumsOrFn = Union[
    GaLoreDimensionNumbers,
    base.Params,
    Callable[[base.Params], Optional[base.Params]],
]

_is_galore_dim_nums = lambda x: isinstance(x, GaLoreDimensionNumbers)
_is_dim_nums_leaf = lambda x: x is None or isinstance(x, GaLoreDimensionNumbers)


def _normalize_axes(
    x: jax.Array, dim_nums: GaLoreDimensionNumbers
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to tuples of non-negative ints."""
  if isinstance(dim_nums.reduction_axis, int):
    reduction_axes = (dim_nums.reduction_axis % x.ndim,)
  else:
    reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

  if isinstance(dim_nums.output_axis, int):
    output_axes = (dim_nums.output_axis % x.ndim,)
  else:
    output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)

  return reduction_axes, output_axes


def _compute_galore_reshape(
    x: jax.Array, dim_nums: GaLoreDimensionNumbers
) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute reshape functions for treating a tensor as a 2D matrix.

  Args:
    x: The tensor to reshape.
    dim_nums: Specification for which axes form the matrix.

  Returns:
    A tuple of (reshape_fn, inverse_fn) where:
    - reshape_fn: transforms x to shape (reduction_size, output_size)
    - inverse_fn: transforms back to original shape
  """
  if x.ndim < 2:
    raise ValueError(
        f"GaLore requires tensors with rank >= 2, got shape {x.shape}"
    )

  reduction_axes, output_axes = _normalize_axes(x, dim_nums)

  if set(reduction_axes) & set(output_axes):
    raise ValueError(
        f"Reduction axes {reduction_axes} and output axes {output_axes} "
        f"must be disjoint. Got dim_nums={dim_nums} for shape {x.shape}"
    )

  # Any axes not in reduction or output are batch axes (should be empty for
  # typical usage, but we handle it for completeness)
  all_specified = set(reduction_axes) | set(output_axes)
  if len(all_specified) != x.ndim:
    raise ValueError(
        f"All axes must be specified. Got reduction={reduction_axes}, "
        f"output={output_axes} for tensor with {x.ndim} dimensions"
    )

  # Compute transpose to put reduction axes first, then output axes
  transpose = reduction_axes + output_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))

  axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  reduction_size = math.prod(axes2shape(reduction_axes))
  output_size = math.prod(axes2shape(output_axes))

  transposed_shape = axes2shape(reduction_axes) + axes2shape(output_axes)

  reshape_fn = lambda y: y.transpose(transpose).reshape(
      reduction_size, output_size
  )
  inverse_fn = lambda y: y.reshape(transposed_shape).transpose(inv_transpose)

  return reshape_fn, inverse_fn


class GaLoreState(NamedTuple):
  """State for the GaLore optimizer.

  Attributes:
    count: Number of update steps taken.
    base_optimizer_state: State for the base optimizer, operating on low-rank
      gradients for 2D params and full gradients for non-2D params.
    projector: Projection matrices for each 2D parameter.

  """

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32
  base_optimizer_state: base.OptState
  projector: base.Updates  # Projection matrices P


def scale_by_galore(
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    base_optimizer: Optional[base.GradientTransformation] = None,
    weight_dimension_numbers: Optional[GaLoreDimNumsOrFn] = None,
) -> base.GradientTransformation:
  """Scale updates using GaLore (Gradient Low-Rank Projection).

  GaLore projects gradients of 2D weight matrices into a low-rank subspace,
  significantly reducing memory for optimizer states while maintaining
  full-parameter learning.

  For tensors that are logically 2D but stored with higher dimensions (e.g.,
  attention projections as [embedding, heads, head_dim]), use
  ``weight_dimension_numbers`` to specify which axes form the matrix.

  .. warning::
    The ``base_optimizer`` must be a **gradient scaling transformation** that
    does NOT require parameter values (e.g.,``scale_by_adam``, ``scale_by_sgd``,
    ``scale_by_lion``). Optimizers that require ``params`` in their update
    function will fail because the base optimizer operates on low-rank shaped
    tensors, not the original parameter shapes.

    **Incompatible optimizers** (will crash or produce incorrect results):

    - ``adamw``, ``lamb``, ``lars``: require params for weight decay or
      trust ratio computation
    - Any optimizer using ``add_decayed_weights`` internally

    **Compatible optimizers**:

    - ``scale_by_adam``, ``scale_by_amsgrad``, ``scale_by_lion``
    - ``scale_by_rms``, ``scale_by_stddev``, ``scale_by_rss``
    - ``sgd`` (with learning_rate=1.0), ``scale_by_schedule``

    For weight decay, use the ``galore`` wrapper with its ``weight_decay``
    parameter, which correctly applies decoupled weight decay in the full
    parameter space.

  Args:
    rank: Target rank for the low-rank projection. Lower rank = less memory
      but potentially slower convergence.
    update_proj_gap: Number of steps between projection matrix updates.
      The projection matrices are recomputed from gradient SVD every this
      many steps.
    scale: Scaling factor applied to the final updates.
    base_optimizer: The base gradient transformation to apply in the low-rank
      subspace for 2D params and full space for non-2D params. Must be a
      gradient-only transformation (see warning above). If None, defaults to
      ``transform.scale_by_adam()``.
    weight_dimension_numbers: Specifies how to treat non-2D tensors as 2D
      matrices. Can be:

      - None: Only project naturally 2D parameters (default behavior)
      - A single ``GaLoreDimensionNumbers``: Apply to all parameters
      - A pytree matching params structure with ``GaLoreDimensionNumbers`` at
        leaves (use None for params to skip)
      - A callable taking params and returning such a pytree

  Returns:
    A GradientTransformation implementing GaLore.

  References:
    Zhao et al., `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
    Projection <https://arxiv.org/abs/2403.03507>`_, 2024
  """
  if base_optimizer is None:
    base_optimizer = transform.scale_by_adam()

  if not isinstance(rank, int):
    raise TypeError(f"`rank` must be an int, got {type(rank)}")
  if rank <= 0:
    raise ValueError(f"`rank` must be positive, got {rank}")

  def _get_dim_nums(params):
    """Resolve dimension numbers for each parameter."""
    if weight_dimension_numbers is None:
      # Default: only 2D params get projected, with standard axes
      return jax.tree.map(
          lambda p: GaLoreDimensionNumbers() if p.ndim == 2 else None, params
      )
    elif callable(weight_dimension_numbers):
      return weight_dimension_numbers(params)
    elif _is_galore_dim_nums(weight_dimension_numbers):
      # Single spec applied to all applicable params
      return jax.tree.map(
          lambda p: weight_dimension_numbers if p.ndim >= 2 else None, params
      )
    else:
      # Already a pytree of dimension numbers
      return weight_dimension_numbers

  def _compute_projection_shapes(p, dim_num):
    """Compute projector and proxy shapes for a parameter."""
    if dim_num is None:
      # No projection for this parameter
      return jnp.zeros((0, 0), dtype=p.dtype), jnp.zeros_like(p)

    # Reshape to 2D for shape computation
    reshape_fn, _ = _compute_galore_reshape(p, dim_num)
    p_2d = reshape_fn(p)
    m_dim, n_dim = p_2d.shape

    use_left = m_dim >= n_dim
    effective_rank = min(rank, m_dim, n_dim)

    if use_left:
      projector_shape = (m_dim, effective_rank)
      proxy_shape = (effective_rank, n_dim)
    else:
      projector_shape = (n_dim, effective_rank)
      proxy_shape = (m_dim, effective_rank)

    projector = jnp.zeros(projector_shape, dtype=p.dtype)
    proxy = jnp.zeros(proxy_shape, dtype=p.dtype)
    return projector, proxy

  def init_fn(params: base.Params) -> GaLoreState:
    # Handle empty trees (e.g., _ParamsPlaceholder from tree_map_params)
    param_leaves, _ = jax.tree.flatten(params)
    if not param_leaves:
      # Empty params - return matching empty state
      base_state = base_optimizer.init(params)
      return GaLoreState(
          count=jnp.zeros([], jnp.int32),
          base_optimizer_state=base_state,
          projector=params,  # Same empty structure
      )

    dim_nums = _get_dim_nums(params)

    # Compute projector and proxy shapes for each parameter
    results = jax.tree.map(
        _compute_projection_shapes,
        params,
        dim_nums,
        is_leaf=_is_dim_nums_leaf,
    )
    projectors, proxies = jax.tree.transpose(
        jax.tree.structure(params),
        jax.tree.structure((0, 0)),
        results,
    )

    # Initialize base optimizer with proxy params (low-rank shaped)
    base_state = base_optimizer.init(proxies)

    return GaLoreState(
        count=jnp.zeros([], jnp.int32),
        base_optimizer_state=base_state,
        projector=projectors,
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
    should_update_proj = (count % update_proj_gap) == 0

    dim_nums = _get_dim_nums(updates)

    def project_to_low_rank(grad, projector, dim_num):
      """Project gradient to low-rank subspace and update projector."""
      if dim_num is None:
        # No projection for this parameter
        return grad, projector

      original_dtype = grad.dtype

      # Reshape to 2D
      reshape_fn, _ = _compute_galore_reshape(grad, dim_num)
      grad_2d = reshape_fn(grad)
      m_dim, n_dim = grad_2d.shape

      effective_rank = min(rank, m_dim, n_dim)
      use_left = m_dim >= n_dim

      if use_left:
        def compute_left_projector():
          grad_f32 = grad_2d.astype(jnp.float32)
          u, _, _ = jnp.linalg.svd(grad_f32, full_matrices=False)
          return u[:, :effective_rank].astype(original_dtype)

        new_projector = jax.lax.cond(
            should_update_proj,
            compute_left_projector,
            lambda: projector,
        )
        low_rank_grad = new_projector.T @ grad_2d
      else:
        def compute_right_projector():
          grad_f32 = grad_2d.astype(jnp.float32)
          _, _, vh = jnp.linalg.svd(grad_f32, full_matrices=False)
          return vh[:effective_rank, :].T.astype(original_dtype)

        new_projector = jax.lax.cond(
            should_update_proj,
            compute_right_projector,
            lambda: projector,
        )
        low_rank_grad = grad_2d @ new_projector

      return low_rank_grad, new_projector

    def project_back_to_full(
        low_rank_update, projector, original_grad, dim_num
    ):
      """Project low-rank update back to full space."""
      if dim_num is None:
        return low_rank_update

      original_dtype = original_grad.dtype

      # Get inverse reshape function
      _, inverse_fn = _compute_galore_reshape(original_grad, dim_num)
      reshape_fn, _ = _compute_galore_reshape(original_grad, dim_num)
      grad_2d = reshape_fn(original_grad)
      m_dim, n_dim = grad_2d.shape
      use_left = m_dim >= n_dim

      if use_left:
        upd_2d = projector @ low_rank_update
      else:
        upd_2d = low_rank_update @ projector.T

      # Reshape back to original shape
      upd = inverse_fn(upd_2d)
      return (scale * upd).astype(original_dtype)

    # Step 1: Project all gradients to low-rank subspace
    projected_results = jax.tree.map(
        project_to_low_rank,
        updates,
        state.projector,
        dim_nums,
        is_leaf=_is_dim_nums_leaf,
    )
    low_rank_grads, new_projectors = jax.tree.transpose(
        jax.tree.structure(updates),
        jax.tree.structure((0, 0)),
        projected_results,
    )

    # Step 2: Apply base optimizer in low-rank space
    low_rank_updates, new_base_state = base_optimizer.update(
        low_rank_grads, state.base_optimizer_state, None
    )

    # Step 3: Project updates back to full space
    full_updates = jax.tree.map(
        project_back_to_full,
        low_rank_updates,
        new_projectors,
        updates,
        dim_nums,
        is_leaf=_is_dim_nums_leaf,
    )

    new_state = GaLoreState(
        count=count_inc,
        base_optimizer_state=new_base_state,
        projector=new_projectors,
    )

    return full_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def galore(
    learning_rate: base.ScalarOrSchedule,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    base_optimizer: Optional[base.GradientTransformation] = None,
    weight_decay: jax.typing.ArrayLike = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    weight_dimension_numbers: Optional[GaLoreDimNumsOrFn] = None,
) -> base.GradientTransformation:
  r"""GaLore: Memory-efficient training via gradient lowrank projection.

  GaLore (Gradient Low-Rank Projection) is a memory-efficient training strategy
  that enables full-parameter learning while reducing optimizer state memory by
  projecting gradients into a low-rank subspace.

  The key insight is that gradients of weight matrices in neural networks often
  exhibit low-rank structure. GaLore exploits this by:

  1. Computing a low-rank projection matrix P using SVD of the gradient
  2. Projecting gradients to a low-rank subspace: R = P^T @ G (or G @ P)
  3. Maintaining optimizer states in the reduced subspace
  4. Projecting updates back to full space: update = P @ normalized_R

  For a weight matrix of shape (m, n) with rank r projection:

  - Standard Adam stores m + v states: 2 * m * n parameters
  - GaLore stores: 2 * min(r*n, m*r) + projection matrix

  This can achieve up to 65% memory reduction for large linear layers.

  .. note::
    GaLore only projects 2D weight matrices by default. Use
    ``weight_dimension_numbers`` to project higher-dimensional tensors
    (like attention projections stored as 3D arrays).

  .. warning::
    The ``base_optimizer`` must be a **gradient scaling transformation** that
    does NOT require parameter values. See ``scale_by_galore`` for details on
    compatible vs incompatible optimizers.

    **Do NOT use**: ``adamw``, ``lamb``, ``lars`` as base_optimizer.

    **Use instead**: ``scale_by_adam``, ``scale_by_lion``, etc., and configure
    weight decay via the ``weight_decay`` parameter of this function.

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
    Objective function: 9.98E+03
    Objective function: 9.96E+03
    Objective function: 9.94E+03
    Objective function: 9.92E+03
    Objective function: 9.90E+03

  Using weight decay (equivalent to AdamW behavior):
    >>> solver = optax.contrib.galore(
    ...     learning_rate=0.01,
    ...     rank=16,
    ...     weight_decay=0.01,  # Use this, NOT adamw as base_optimizer
    ... )

  Using a custom base optimizer:
    >>> solver = optax.contrib.galore(
    ...     learning_rate=0.01,
    ...     rank=16,
    ...     base_optimizer=optax.scale_by_adam(b1=0.9, b2=0.99),
    ... )

  Projecting 3D attention weights as 2D matrices:
    >>> from optax.contrib import GaLoreDimensionNumbers
    >>> # For attention weights shaped (embed_dim, num_heads, head_dim)
    >>> dim_nums = {'attn': GaLoreDimensionNumbers(
    ...     reduction_axis=0,      # embed_dim
    ...     output_axis=(1, 2),    # heads*head_dim
    ... )}
    >>> solver = optax.contrib.galore(
    ...     learning_rate=0.01, rank=16, weight_dimension_numbers=dim_nums
    ... )

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    rank: Target rank for the low-rank projection. Lower values save more
      memory but may slow convergence. Default 128 is a good starting point.
    update_proj_gap: Number of steps between projection matrix updates.
      The projectors are recomputed from the gradient SVD every this many
      steps to adapt to the changing gradient landscape.
    scale: Additional scaling factor for updates.
    base_optimizer: The base gradient transformation to apply in the low-rank
      subspace. Must be a gradient-only transformation like ``scale_by_adam``,
      NOT an optimizer requiring params like ``adamw``. If None, defaults to
      ``optax.scale_by_adam()``. If the base optimizer includes a learning
      rate, set ``learning_rate=1.0`` here to avoid double-scaling.
    weight_decay: Strength of decoupled weight decay regularization (as in
      AdamW). This is applied correctly in full parameter space, unlike
      weight decay in the base optimizer which would fail.
    mask: A tree with same structure as params PyTree, or a Callable that
      returns such a pytree. Leaves should be booleans indicating whether
      to apply weight decay to each parameter.
    weight_dimension_numbers: Specifies how to treat non-2D tensors as 2D
      matrices for projection. See ``scale_by_galore`` for details.

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
          base_optimizer=base_optimizer,
          weight_dimension_numbers=weight_dimension_numbers,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )
