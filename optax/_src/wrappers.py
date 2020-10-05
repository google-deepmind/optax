# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Transformation wrappers."""

from typing import Any, NamedTuple

from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
import numpy as np

from optax._src import transform


def flatten(
    inner: transform.GradientTransformation
) -> transform.GradientTransformation:
  """Flattens parameters and gradients for init and update of inner transform.

  This can reduce the overhead of performing many calculations on lots of small
  variables, at the cost of slightly increased memory usage.

  Args:
    inner: Inner transformation to flatten inputs for.

  Returns:
    New GradientTransformation.
  """

  def _flatten(params):
    """Flattens and concatenates all tensors in params to a single vector."""
    params, _ = tree_flatten(params)
    return jnp.concatenate([jnp.reshape(param, [-1]) for param in params])

  def _unflatten(updates, flat):
    """Extracts tensors from flat, using the structure and shapes of params."""
    updates_flat, treedef = tree_flatten(updates)
    offsets = []
    for update in updates_flat:
      size = np.prod(update.shape)
      if offsets:
        offsets.append(size + offsets[-1])
      else:
        offsets.append(size)
    del offsets[-1]
    flat_split = jnp.split(flat, offsets)
    reshaped = [
        jnp.reshape(flat_update, update.shape)
        for flat_update, update in zip(flat_split, updates_flat)
    ]
    return tree_unflatten(treedef, reshaped)

  def init_fn(params):
    flat = _flatten(params)
    return inner.init(flat)

  def update_fn(updates, state, params=None):
    if params is not None:
      params = _flatten(params)
    updates_flat, state = inner.update(_flatten(updates), state, params)
    updates = _unflatten(updates, updates_flat)
    return updates, state

  return transform.GradientTransformation(init_fn, update_fn)


class ApplyIfFiniteState(NamedTuple):
  """State of the `GradientTransformation` returned by `apply_if_finite`.

  Fields:
    notfinite_count: Number of consecutive gradient updates containing an Inf or
      a NaN. This number is reset to 0 whenever a gradient update without an Inf
      or a NaN is done.
    last_finite: Whether or not the last gradient update contained an Inf of a
      NaN.
    total_notfinite: Total number of gradient updates containing an Inf or
      a NaN since this optimiser was initialised. This number is never reset.
    inner_state: The state of the inner `GradientTransformation`.
  """
  notfinite_count: jnp.array
  last_finite: jnp.array
  total_notfinite: jnp.array
  inner_state: Any


def apply_if_finite(
    inner: transform.GradientTransformation,
    max_consecutive_errors: int
) -> transform.GradientTransformation:
  """A function that wraps an optimiser to make it robust to a few NaNs or Infs.

  The purpose of this function is to prevent any optimisation to happen if the
  gradients contain NaNs or Infs. That is, when a NaN of Inf is detected in the
  gradients, the wrapped optimiser ignores that gradient update. If the NaNs or
  Infs persist after a given number of updates, the wrapped optimiser gives up
  and accepts the update.

  Args:
    inner: Inner transformation to be wrapped.
    max_consecutive_errors: Maximum number of consecutive gradient updates
      containing NaNs of Infs that the wrapped optimiser will ignore. After
      that many ignored updates, the optimiser will give up and accept.

  Returns:
    New GradientTransformation.
  """

  def init(params):
    return ApplyIfFiniteState(
        notfinite_count=jnp.zeros([], jnp.int64),
        last_finite=jnp.array(True, jnp.bool_),
        total_notfinite=jnp.zeros([], jnp.int64),
        inner_state=inner.init(params))

  def update(grads, state, params=None):
    inner_state = state.inner_state
    flat_grads = tree_flatten(grads)[0]
    isfinite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_grads]))
    notfinite_count = jnp.where(isfinite, jnp.zeros([], jnp.int64),
                                1 + state.notfinite_count)

    def do_update(_):
      return inner.update(grads, inner_state, params)
    def reject_update(_):
      return (tree_map(jnp.zeros_like, grads), inner_state)

    updates, new_inner_state = lax.cond(
        jnp.logical_or(isfinite, notfinite_count > max_consecutive_errors),
        do_update, reject_update, operand=None)

    return updates, ApplyIfFiniteState(
        notfinite_count=notfinite_count,
        last_finite=isfinite,
        total_notfinite=jnp.logical_not(isfinite) + state.total_notfinite,
        inner_state=new_inner_state)

  return transform.GradientTransformation(init=init, update=update)
