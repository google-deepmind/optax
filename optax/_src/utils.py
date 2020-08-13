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
import numpy as np


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
    sum_axis = [-(i +1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob, axis=sum_axis)

  @property
  def log_scale(self):
    return self._log_scale

  @property
  def params(self):
    return [self._mean, self._log_scale]


def rademacher(key, shape, dtype=np.int32):
  """Sample from a Rademacher distribution.

  Args:
    key: a PRNGKey key.
    shape: The shape of the returned samples.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples, of shape `shape`. Each element in the output has
    a 50% change of being 1 or -1.

  """
  random_bernoulli = jax.random.categorical(
      key=key, logits=jnp.array([0.5, 0.5]), shape=shape)
  return (2 * random_bernoulli - 1).astype(dtype)


def one_sided_maxwell(key, shape=(), dtype=jnp.float32):
  """Sample from a one sided Maxwell distribution.

  The scipy counterpart is `scipy.stats.maxwell`.

  Args:
    key: a PRNGKey key.
    shape: The shape of the returned samples.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples, of shape `shape`.

  """
  # Generate samples using:
  # sqrt(X^2 + Y^2 + Z^2), X,Y,Z ~N(0,1)
  shape = shape + (3,)
  norm_rvs = jax.random.normal(key=key, shape=shape, dtype=dtype)
  maxwell_rvs = jnp.linalg.norm(norm_rvs, axis=-1)
  return maxwell_rvs


def double_sided_maxwell(key, loc, scale, shape=(), dtype=jnp.float32):
  """Sample from a double sided Maxwell distribution.

  Args:
    key: a PRNGKey key.
    loc: The location parameter of the distribution.
    scale: The scale parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """

  # mu + sigma* sgn(U-0.5)* one_sided_maxwell U~Unif;
  params_shapes = jax.lax.broadcast_shapes(np.shape(loc), np.shape(scale))
  if not shape:
    shape = params_shapes

  shape = shape + params_shapes
  maxwell_key, rademacher_key = jax.random.split(key)
  maxwell_rvs = one_sided_maxwell(maxwell_key, shape=shape, dtype=dtype)
  # Generate random signs for the symmetric variates.
  random_sign = rademacher(rademacher_key, shape=shape, dtype=dtype)
  assert random_sign.shape == maxwell_rvs.shape

  samples = random_sign * maxwell_rvs * scale + loc
  return samples


def weibull_min(key, scale, concentration, shape=(), dtype=jnp.float32):
  """Sample from a Weibull distribution.

  The scipy counterpart is `scipy.stats.weibull_min`.

  Args:
    key: a PRNGKey key.
    scale: The scale parameter of the distribution.
    concentration: The concentration parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """
  random_uniform = jax.random.uniform(
      key=key, shape=shape, minval=0, maxval=1, dtype=dtype)

  # Inverse weibull CDF.
  return jnp.power(-jnp.log1p(-random_uniform), 1.0/concentration) * scale
