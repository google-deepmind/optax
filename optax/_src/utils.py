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
"""Utility functions for testing."""

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as multivariate_normal

from optax._src import base


def tile_second_to_last_dim(a):
  ones = jnp.ones_like(a)
  a = jnp.expand_dims(a, axis=-1)
  return jnp.expand_dims(ones, axis=-2) * a


def global_norm(updates: base.Updates) -> base.Updates:
  """Compute the global norm across a nested structure of tensors."""
  return jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def safe_int32_increment(count):
  """Increments int32 counter by one.

  Normally `max_int + 1` would overflow to `min_int`. This functions ensures
  that when `max_int` is reached the counter stays at `max_int`.

  Args:
    count: a counter to be incremented.

  Returns:
    a counter incremented by 1, or max_int if the maximum precision is reached.
  """
  chex.assert_type(count, jnp.int32)
  max_int32_value = jnp.iinfo(jnp.int32).max
  one = jnp.array(1, dtype=jnp.int32)
  return jnp.where(count < max_int32_value, count + one, max_int32_value)


def safe_norm(x, min_norm):
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

  The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
  because jax will evaluate both branches of the `jnp.maximum`. This function
  will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
  """
  norm = jnp.linalg.norm(x)
  x = jnp.where(norm < min_norm, jnp.ones_like(x), x)
  return jnp.where(norm < min_norm, min_norm, jnp.linalg.norm(x))


def merge_small_dims(shape_to_merge, max_dim):
  """Merge small dimensions.

  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]

  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.

  Returns:
    Merged shape.
  """
  resulting_shape = []
  product = 1
  for d in shape_to_merge:
    if product * d <= max_dim:
      product *= d
    else:
      if product > 1:
        resulting_shape.append(product)
      product = d
  if product > 1:
    resulting_shape.append(product)
  return resulting_shape


def pad_matrix(mat, max_size):
  """Pad a matrix to a max_size.

  Args:
    mat: a matrix to pad.
    max_size: matrix size requested.

  Returns:
    Given M returns [[M, 0], [0, I]]
  """
  size = mat.shape[0]
  assert size <= max_size
  if size == max_size:
    return mat
  pad_size = max_size - size
  zs1 = jnp.zeros([size, pad_size], dtype=mat.dtype)
  zs2 = jnp.zeros([pad_size, size], dtype=mat.dtype)
  eye = jnp.eye(pad_size, dtype=mat.dtype)
  mat = jnp.concatenate([mat, zs1], 1)
  mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
  return mat


def efficient_cond(predicate, compute_fn, init_state, *args, **kwargs):
  """Avoids wasteful buffer allocation with XLA."""

  def _iter_body(unused_state):
    results = compute_fn(*args, **kwargs)
    return tuple([False] + list(results))

  def _iter_condition(state):
    return state[0]

  results = jax.lax.while_loop(
      _iter_condition, _iter_body, tuple([predicate] + init_state))
  return tuple(results[1:])


def set_diags(a, new_diags):
  """Set the diagonals of every DxD matrix in an input of shape NxDxD.

  Args:
    a: rank 3, tensor NxDxD.
    new_diags: NxD matrix, the new diagonals of each DxD matrix.

  Returns:
    NxDxD tensor, with the same contents as `a` but with the diagonal
      changed to `new_diags`.
  """
  n, d, d1 = a.shape
  assert d == d1

  indices1 = jnp.repeat(jnp.arange(n), d)
  indices2 = jnp.tile(jnp.arange(d), n)
  indices3 = indices2

  # Use numpy array setting
  a = a.at[indices1, indices2, indices3].set(new_diags.flatten())
  return a


def multi_normal(loc, log_scale):
  return MultiNormalDiagFromLogScale(loc=loc, log_scale=log_scale)


class MultiNormalDiagFromLogScale():
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc, log_scale):
    self._log_scale = log_scale
    self._scale = jnp.exp(log_scale)
    self._mean = loc
    self._param_shape = jax.lax.broadcast_shapes(
        self._mean.shape, self._scale.shape)

  def sample(self, shape, seed):
    sample_shape = shape + self._param_shape
    return jax.random.normal(
        seed, shape=sample_shape) * self._scale  + self._mean

  def log_prob(self, x):
    log_prob = multivariate_normal.logpdf(x, loc=self._mean, scale=self._scale)
    # Sum over parameter axes.
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob, axis=sum_axis)

  @property
  def log_scale(self):
    return self._log_scale

  @property
  def params(self):
    return [self._mean, self._log_scale]
