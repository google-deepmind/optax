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

from typing import Optional, Tuple, Sequence

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as multivariate_normal

from optax._src import linear_algebra
from optax._src import numerics


def tile_second_to_last_dim(a: chex.Array) -> chex.Array:
  ones = jnp.ones_like(a)
  a = jnp.expand_dims(a, axis=-1)
  return jnp.expand_dims(ones, axis=-2) * a


def canonicalize_dtype(
    dtype: Optional[chex.ArrayDType]) -> Optional[chex.ArrayDType]:
  """Canonicalise a dtype, skip if None."""
  if dtype is not None:
    return jax.dtypes.canonicalize_dtype(dtype)
  return dtype


def cast_tree(tree: chex.ArrayTree,
              dtype: Optional[chex.ArrayDType]) -> chex.ArrayTree:
  """Cast tree to given dtype, skip if None."""
  if dtype is not None:
    return jax.tree_util.tree_map(lambda t: t.astype(dtype), tree)
  else:
    return tree


def set_diags(a: chex.Array, new_diags: chex.Array) -> chex.Array:
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


class MultiNormalDiagFromLogScale():
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc: chex.Array, log_scale: chex.Array):
    self._log_scale = log_scale
    self._scale = jnp.exp(log_scale)
    self._mean = loc
    self._param_shape = jax.lax.broadcast_shapes(
        self._mean.shape, self._scale.shape)

  def sample(self, shape: Sequence[int],
             seed: chex.PRNGKey) -> chex.Array:
    sample_shape = tuple(shape) + self._param_shape
    return jax.random.normal(
        seed, shape=sample_shape) * self._scale  + self._mean

  def log_prob(self, x: chex.Array) -> chex.Array:
    log_prob = multivariate_normal.logpdf(x, loc=self._mean, scale=self._scale)
    # Sum over parameter axes.
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob, axis=sum_axis)

  @property
  def log_scale(self) -> chex.Array:
    return self._log_scale

  @property
  def params(self) -> Sequence[chex.Array]:
    return [self._mean, self._log_scale]


def multi_normal(loc: chex.Array,
                 log_scale: chex.Array) -> MultiNormalDiagFromLogScale:
  return MultiNormalDiagFromLogScale(loc=loc, log_scale=log_scale)


@jax.custom_vjp
def _scale_gradient(inputs: chex.ArrayTree, scale: float) -> chex.ArrayTree:
  """Internal gradient scaling implementation."""
  del scale  # Only used for the backward pass defined in _scale_gradient_bwd.
  return inputs


def _scale_gradient_fwd(inputs: chex.ArrayTree,
                        scale: float) -> Tuple[chex.ArrayTree, float]:
  return _scale_gradient(inputs, scale), scale


def _scale_gradient_bwd(scale: float,
                        g: chex.ArrayTree) -> Tuple[chex.ArrayTree, None]:
  return (jax.tree_util.tree_map(lambda g_: g_ * scale, g), None)


_scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)


def scale_gradient(inputs: chex.ArrayTree, scale: float) -> chex.ArrayTree:
  """Scales gradients for the backwards pass.

  Args:
    inputs: A nested array.
    scale: The scale factor for the gradient on the backwards pass.

  Returns:
    An array of the same structure as `inputs`, with scaled backward gradient.
  """
  # Special case scales of 1. and 0. for more efficiency.
  if scale == 1.:
    return inputs
  elif scale == 0.:
    return jax.lax.stop_gradient(inputs)
  else:
    return _scale_gradient(inputs, scale)


# TODO(b/183800387): remove legacy aliases.
safe_norm = numerics.safe_norm
safe_int32_increment = numerics.safe_int32_increment
global_norm = linear_algebra.global_norm
