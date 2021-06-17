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
"""Factorized optimizers."""

from typing import Any, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import base
from optax._src import utils

# pylint:disable=no-value-for-parameter


def _decay_rate_pow(i: int, exponent: float = 0.8) -> float:
  """Second-order moment decay schedule."""
  t = jnp.array(i, jnp.float32) + 1.0
  return 1.0 - t**(-exponent)


def _factored_dims(
    shape: base.Shape,
    factored: bool,
    min_dim_size_to_factor: int
) -> Optional[Tuple[int, int]]:
  """Whether to use a factored second moment estimator.

  This function returns a tuple with the two largest axes to reduce over.
  If no two dimensions have size >= min_dim_size_to_factor, return None.

  Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for 2d vars.
    min_dim_size_to_factor: only factor accumulator if two array dimensions
        have at least this size.

  Returns:
    None or a tuple of ints
  """
  if not factored or len(shape) < 2:
    return None
  sorted_dims = np.argsort(shape)
  if shape[sorted_dims[-2]] < min_dim_size_to_factor:
    return None
  return int(sorted_dims[-2]), int(sorted_dims[-1])


class FactoredParameterStats(NamedTuple):
  """Stats associated to each parameter of the model."""
  v_row: chex.Array  # used for factored params.
  v_col: chex.Array  # used for factored params.
  v: chex.Array  # used for params where factoring is skipped.


class FactoredState(base.OptState):
  """Overall state of the gradient transformation."""
  count: chex.Array  # number of update steps.
  stats: Any  # Statistics held for each parameter.


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon: float = 1e-30):
  """Scaling by a factored estimate of the gradient rms (as in Adafactor).

  This is a so-called "1+epsilon" scaling algorithms, that is extremely memory
  efficient compared to RMSProp/Adam, and has had wide success when applied to
  large-scale training of attention-based models.

  References:
    [Shazeer et al, 2018](https://arxiv.org/abs/1804.04235)

  Args:
      factored: boolean: whether to use factored second-moment estimates..
      decay_rate: float: controls second-moment exponential decay schedule.
      step_offset: for finetuning, one may set this to the starting step-number
        of the fine tuning phase.
      min_dim_size_to_factor: only factor accumulator if two array dimensions
        are at least this size.
      epsilon: Regularization constant for squared gradient.

  Returns:
    the corresponding `GradientTransformation`.
  """

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      shape = param.shape
      stats = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v']}
      factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        vr_shape = np.delete(shape, d0)
        vc_shape = np.delete(shape, d1)
        stats['v_row'] = jnp.zeros(vr_shape, dtype=jnp.float32)
        stats['v_col'] = jnp.zeros(vc_shape, dtype=jnp.float32)
      else:
        stats['v'] = jnp.zeros(param.shape, dtype=jnp.float32)
      return FactoredParameterStats(**stats)

    return FactoredState(
        count=jnp.zeros([], jnp.int32),
        stats=jax.tree_map(_init, params))

  def update_fn(grads, state, params):
    """Apply gradient transformation."""

    def _update(grad, stats, param, step):
      shape = param.shape
      grad = grad.astype(jnp.float32)
      decay_rate_t = _decay_rate_pow(step - step_offset, decay_rate)

      # Scaled by factorized second moment statistics.
      new_stats = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v']}
      factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        grad_sqr = grad * grad + epsilon
        new_v_row = (
            decay_rate_t * stats.v_row +
            (1. - decay_rate_t) * jnp.mean(grad_sqr, axis=d0))
        new_v_col = (
            decay_rate_t * stats.v_col +
            (1. - decay_rate_t) * jnp.mean(grad_sqr, axis=d1))
        new_stats['v_row'] = new_v_row
        new_stats['v_col'] = new_v_col
        reduced_d1 = d1-1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
        row_factor = (new_v_row / row_col_mean) ** -0.5
        col_factor = (new_v_col) ** -0.5
        update = (
            grad *
            jnp.expand_dims(row_factor, axis=d0) *
            jnp.expand_dims(col_factor, axis=d1))
      else:
        grad_sqr = grad * grad + epsilon
        new_v = decay_rate_t * stats.v + (1. - decay_rate_t) * grad_sqr
        new_stats['v'] = new_v
        update = grad * (new_v)**-0.5

      return update, FactoredParameterStats(**new_stats)

    # Transform grad and compute new per-parameter stats.
    output = jax.tree_multimap(
        lambda g, s, p: _update(g, s, p, state.count),
        grads, state.stats, params)

    # Unpack updates / stats and return.
    treedef = jax.tree_structure(grads)
    updates_flat, stats_flat = list(zip(*treedef.flatten_up_to(output)))
    updates = jax.tree_unflatten(treedef, updates_flat)
    stats = jax.tree_unflatten(treedef, stats_flat)
    return updates, FactoredState(
        count=utils.safe_int32_increment(state.count),
        stats=stats)

  return base.GradientTransformation(init_fn, update_fn)
