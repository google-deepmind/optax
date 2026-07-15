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
"""Experimental sharding utilities for Optax gradient transformations.

This module provides utilities for zero-redundancy sharding of Optax optimizer
state. The core idea is to shard optimizer state across more mesh axes than the
model parameters, reducing per-device memory usage of the optimizer without
changing the shapes of the state arrays.

This module draws on ideas from ``jax_privacy.sharding_utils``, adapting them
for use with arbitrary Optax gradient transformations.

.. admonition:: Assumptions

   1. **Explicit sharding API required.** This module assumes that the calling
      program uses JAX's explicit sharding API (i.e., "sharding and types"), as
      described at
      https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html.
      In particular, a mesh should be set via ``jax.sharding.set_mesh()`` and
      arrays should carry type-level sharding information.

   2. **Performance characteristics not yet evaluated.** While we provide test
      coverage ensuring that the shardings of intermediate optimizer state
      arrays work as intended, we have **not** yet evaluated the performance
      characteristics of these APIs. If you observe unexpected performance
      behaviour (e.g., slow compilation, excessive cross-device communication,
      or elevated memory usage), please raise an issue on GitHub.
"""

import math
from typing import Any, cast

import jax
from optax._src import base

P = jax.sharding.PartitionSpec


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_explicit_mesh(mesh: jax.sharding.Mesh) -> None:
  """Raise if any mesh axis does not have ``AxisType.Explicit``."""
  if not all(
      axis_type == jax.sharding.AxisType.Explicit
      for axis_type in mesh.axis_types
  ):
    raise RuntimeError(
        'with_custom_sharding requires an explicit mesh. Please set the mesh '
        'using jax.sharding.set_mesh() with '
        'axis_types=jax.sharding.AxisType.Explicit.'
    )


def _get_mesh(*pytrees: Any) -> jax.sharding.Mesh:
  """Extract the mesh from the first leaf with a ``NamedSharding``."""
  for pytree in pytrees:
    for leaf in jax.tree.leaves(pytree):
      sharding = jax.typeof(leaf).sharding
      if isinstance(sharding, jax.sharding.NamedSharding):
        return cast(jax.sharding.Mesh, sharding.mesh)
  raise ValueError(
      'Could not extract mesh from any leaf. Ensure arrays carry type-level '
      'sharding information (see jax.sharding.set_mesh()).'
  )


def _to_struct(leaf: jax.Array) -> jax.ShapeDtypeStruct:
  """Convert a concrete array to its abstract ``ShapeDtypeStruct``."""
  typ = jax.typeof(leaf)
  return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=typ.sharding)


def _maybe_reshard(leaf: jax.Array, abstract: jax.ShapeDtypeStruct):
  """Reshard *leaf* to match *abstract*'s sharding, if it has one."""
  return jax.reshard(leaf, abstract.sharding) if abstract.sharding else leaf


def _reshard_to_abstract(pytree: Any, abstract_pytree: Any) -> Any:
  """Reshard each leaf of *pytree* to match shardings in *abstract_pytree*."""
  return jax.tree.map(_maybe_reshard, pytree, abstract_pytree)


def _reshard_leaves_enhanced(pytree: Any) -> Any:
  """Reshard every leaf of *pytree* to its enhanced sharding."""
  enhanced_abstract = _enhance_abstract_state(jax.tree.map(_to_struct, pytree))
  return _reshard_to_abstract(pytree, enhanced_abstract)


def _enhance_abstract_state(abstract_state: Any) -> Any:
  """Map abstract optimizer state to one with enhanced sharding annotations."""

  def _enhance_leaf(leaf):
    if not isinstance(leaf.sharding, jax.sharding.NamedSharding):
      return leaf
    enhanced_pspec = _compute_enhanced_pspec(leaf)
    mesh = cast(jax.sharding.Mesh, leaf.sharding.mesh)
    return jax.ShapeDtypeStruct(
        leaf.shape,
        leaf.dtype,
        sharding=jax.sharding.NamedSharding(mesh, enhanced_pspec),
    )

  return jax.tree.map(_enhance_leaf, abstract_state)


def _compute_enhanced_pspec(
    abstract_array: jax.ShapeDtypeStruct,
) -> jax.sharding.PartitionSpec:
  """Compute an enhanced PartitionSpec using unused mesh axes."""
  # Greedy algorithm: iterate over unused mesh axes in decreasing order of
  # size and assign each to the largest array dimension that is evenly
  # divisible by the cumulative shard size. Returns a PartitionSpec that
  # utilises as many mesh axes as possible without changing the array shape.
  shape = abstract_array.shape
  if not shape:
    # Scalar: nothing to shard.
    return P()

  sharding = abstract_array.sharding
  if isinstance(sharding, jax.sharding.NamedSharding):
    current_pspec = sharding.spec
    mesh = cast(jax.sharding.Mesh, sharding.mesh)
  else:
    raise TypeError(
        'compute_enhanced_pspec requires a NamedSharding, got '
        f'{type(sharding)}.'
    )

  ndim = len(shape)

  # Parse current pspec into per-dimension axis lists.
  dim_axes: list[list[str]] = [[] for _ in range(ndim)]
  used_axes: set[str] = set()

  for i, entry in enumerate(current_pspec):
    if i >= ndim:
      break
    if entry is None:
      continue
    elif isinstance(entry, str):
      dim_axes[i].append(entry)
      used_axes.add(entry)
    elif isinstance(entry, tuple):
      for ax in entry:
        dim_axes[i].append(ax)
        used_axes.add(ax)

  # Unused mesh axes, sorted by size descending (greedy: largest first).
  unused_axes = sorted(
      (
          (name, mesh.shape[name])
          for name in mesh.axis_names
          if name not in used_axes
      ),
      key=lambda pair: pair[1],
      reverse=True,
  )

  # Greedy assignment: for each unused axis, assign to the largest compatible
  # dimension.
  for ax_name, ax_size in unused_axes:
    best_dim = None
    best_dim_size = -1
    for i in range(ndim):
      current_shard_size = (
          math.prod(mesh.shape[a] for a in dim_axes[i]) if dim_axes[i] else 1
      )
      if shape[i] % (current_shard_size * ax_size) == 0:
        if shape[i] > best_dim_size:
          best_dim = i
          best_dim_size = shape[i]
    if best_dim is not None:
      dim_axes[best_dim].append(ax_name)

  # Build the resulting PartitionSpec.
  entries: list[str | tuple[str, ...] | None] = []
  for axes in dim_axes:
    if not axes:
      entries.append(None)
    elif len(axes) == 1:
      entries.append(axes[0])
    else:
      entries.append(tuple(axes))
  return P(*entries)


# ---------------------------------------------------------------------------
# Public: with_custom_sharding wrapper
# ---------------------------------------------------------------------------


def with_custom_sharding(
    inner: base.GradientTransformation,
) -> base.GradientTransformation:
  """Wrap a gradient transformation with zero-redundancy state sharding.

  This wrapper modifies an existing Optax :class:`GradientTransformation` so
  that its optimizer state is sharded across *more* mesh axes than the model
  parameters. This reduces per-device memory usage of the optimizer state at
  the cost of additional resharding operations during the ``update`` step.

  Unlike the flattening approach in ``jax_privacy.sharding_utils``, this
  wrapper **preserves the shapes** of all optimizer-state arrays and only
  modifies their shardings. A greedy algorithm (see
  :func:`compute_enhanced_pspec`) assigns unused mesh axes to array dimensions
  wherever the dimension size is evenly divisible by the mesh-axis size.

  Example usage::

    import optax
    from optax.experimental import sharding
    tx = sharding.with_custom_sharding(optax.adam(1e-3))

  Args:
    inner: The base gradient transformation to wrap.

  Returns:
    A new :class:`GradientTransformation` whose optimizer state uses enhanced
    (zero-redundancy) sharding.
  """

  def init_fn(params):
    # Extract mesh from params' type-level sharding info.
    mesh = _get_mesh(params)
    _check_explicit_mesh(mesh)

    # Materialise the optimizer state, then reshard to enhanced shardings.
    state = inner.init(params)
    enhanced_abstract = _enhance_abstract_state(jax.tree.map(_to_struct, state))
    return _reshard_to_abstract(state, enhanced_abstract)

  def update_fn(updates, state, params=None):
    # Reshard updates (and params, if given) into the enhanced sharding domain.
    enhanced_updates = _reshard_leaves_enhanced(updates)
    enhanced_params = (
        _reshard_leaves_enhanced(params) if params is not None else None
    )

    # Delegate to the inner transform.
    new_updates, new_state = inner.update(
        enhanced_updates,
        state,
        enhanced_params,
    )

    return new_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)
