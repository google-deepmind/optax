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
"""Utility functions for testing."""

import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as multivariate_normal


def tile_second_to_last_dim(a):
  ones = jnp.ones_like(a)
  a = jnp.expand_dims(a, axis=-1)

  return jnp.expand_dims(ones, axis=-2) * a


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
