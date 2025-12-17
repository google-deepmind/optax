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

import contextlib
from typing import Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as multivariate_normal
import numpy as np
from optax._src import base
from optax._src import numerics


def tile_second_to_last_dim(a: jax.typing.ArrayLike) -> jax.Array:
  ones = jnp.ones_like(a)
  a = jnp.expand_dims(a, axis=-1)
  return jnp.expand_dims(ones, axis=-2) * a


def canonicalize_dtype(
    dtype: Optional[jax.typing.DTypeLike],
) -> Optional[jax.typing.DTypeLike]:
  """Canonicalise a dtype, skip if None."""
  if dtype is not None:
    return jax.dtypes.canonicalize_dtype(dtype)
  return dtype


def canonicalize_key(key_or_seed: jax.Array | int) -> jax.Array:
  """Canonicalize a random key or an int representing a seed to a random key."""
  if (isinstance(key_or_seed, jax.Array) and jnp.issubdtype(
      key_or_seed.dtype, jax.dtypes.prng_key
  )):
    return key_or_seed
  return jax.random.key(key_or_seed)


def check_subdtype(array: jax.typing.ArrayLike, subdtype: jax.typing.DTypeLike):
  """Check that `array`'s dtype is a subdtype of `subdtype`."""
  dtype = jax.dtypes.result_type(array)
  if not jnp.issubdtype(dtype, subdtype):
    raise TypeError(
        f'Expected the input to have a dtype that is a subdtype of {subdtype}, '
        f'got {dtype} instead'
    )


def check_shapes_equal(a: jax.typing.ArrayLike, b: jax.typing.ArrayLike):
  """Check that `a` and `b` have the same shape."""
  a_shape = a.shape if hasattr(a, 'shape') else np.asarray(a).shape
  b_shape = b.shape if hasattr(b, 'shape') else np.asarray(b).shape
  if a_shape != b_shape:
    raise ValueError(f'Shape mismatch: got {a_shape} and {b_shape}.')


def check_rank(array: jax.typing.ArrayLike, rank: int):
  """Check that `array` has the specified rank."""
  shape = array.shape if hasattr(array, 'shape') else np.asarray(array).shape
  array_rank = len(shape)
  if array_rank != rank:
    raise ValueError(
        f'Expected the input to have rank {rank}, got {array_rank} instead'
    )


def set_diags(a: jax.Array, new_diags: jax.typing.ArrayLike) -> jax.Array:
  """Set the diagonals of every DxD matrix in an input of shape NxDxD.

  Args:
    a: rank 3, tensor NxDxD.
    new_diags: NxD matrix, the new diagonals of each DxD matrix.

  Returns:
    NxDxD tensor, with the same contents as `a` but with the diagonal
      changed to `new_diags`.
  """
  a_dim, new_diags_dim = len(a.shape), len(new_diags.shape)
  if a_dim != 3:
    raise ValueError(f'Expected `a` to be a 3D tensor, got {a_dim}D instead')
  if new_diags_dim != 2:
    raise ValueError(
        f'Expected `new_diags` to be a 2D array, got {new_diags_dim}D instead'
    )
  n, d, d1 = a.shape
  n_diags, d_diags = new_diags.shape
  if d != d1:
    raise ValueError(
        f'Shape mismatch: expected `a.shape` to be {(n, d, d)}, '
        f'got {(n, d, d1)} instead'
    )
  if d_diags != d or n_diags != n:
    raise ValueError(
        f'Shape mismatch: expected `new_diags.shape` to be {(n, d)}, '
        f'got {(n_diags, d_diags)} instead'
    )

  indices1 = jnp.repeat(jnp.arange(n), d)
  indices2 = jnp.tile(jnp.arange(d), n)
  indices3 = indices2

  # Use numpy array setting
  a = a.at[indices1, indices2, indices3].set(new_diags.flatten())
  return a


class MultiNormalDiagFromLogScale:
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(
      self, loc: jax.typing.ArrayLike, log_scale: jax.typing.ArrayLike):
    self._log_scale = jnp.asarray(log_scale)
    self._scale = jnp.exp(log_scale)
    self._mean = jnp.asarray(loc)
    self._param_shape = jax.lax.broadcast_shapes(
        self._mean.shape, self._scale.shape
    )

  def sample(self, shape: base.Shape, key: base.PRNGKey) -> jax.Array:
    sample_shape = tuple(shape) + self._param_shape
    return (
        jax.random.normal(key, shape=sample_shape) * self._scale + self._mean
    )

  def log_prob(self, x: jax.typing.ArrayLike) -> jax.Array:
    log_prob = multivariate_normal.logpdf(x, loc=self._mean, scale=self._scale)
    # Sum over parameter axes.
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob, axis=sum_axis)

  @property
  def log_scale(self) -> jax.Array:
    return self._log_scale

  @property
  def params(self) -> Sequence[jax.Array]:
    return [self._mean, self._log_scale]


def multi_normal(
    loc: jax.typing.ArrayLike, log_scale: jax.typing.ArrayLike
) -> MultiNormalDiagFromLogScale:
  return MultiNormalDiagFromLogScale(loc=loc, log_scale=log_scale)


@jax.custom_vjp
def _scale_gradient(
    inputs: chex.ArrayTree, scale: jax.typing.ArrayLike) -> chex.ArrayTree:
  """Internal gradient scaling implementation."""
  del scale  # Only used for the backward pass defined in _scale_gradient_bwd.
  return inputs


def _scale_gradient_fwd(
    inputs: chex.ArrayTree, scale: jax.typing.ArrayLike
) -> tuple[chex.ArrayTree, jax.typing.ArrayLike]:
  return _scale_gradient(inputs, scale), scale


def _scale_gradient_bwd(
    scale: jax.typing.ArrayLike, g: chex.ArrayTree
) -> tuple[chex.ArrayTree, None]:
  return (jax.tree.map(lambda g_: g_ * scale, g), None)


_scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)


def scale_gradient(
    inputs: chex.ArrayTree, scale: jax.typing.ArrayLike) -> chex.ArrayTree:
  """Scales gradients for the backwards pass.

  Args:
    inputs: A nested array.
    scale: The scale factor for the gradient on the backwards pass.

  Returns:
    An array of the same structure as `inputs`, with scaled backward gradient.
  """
  # Special case scales of 1. and 0. for more efficiency.
  if scale == 1.0:
    return inputs
  if scale == 0.0:
    return jax.lax.stop_gradient(inputs)
  return _scale_gradient(inputs, scale)


@contextlib.contextmanager
def x64_precision(enable_x64_precision: bool = True):
  """Context manager to temporarily enable x64 precision.

  Args:
    enable_x64_precision: Whether to enable or disable x64 precision within the
      context.

  Yields:
    None

  Examples:
    >>> from optax._src.utils import x64_precision
    >>> with x64_precision(enable_x64_precision=True):
    ...   print(jnp.float64(1.0).dtype.name)
    float64
    >>> with x64_precision(enable_x64_precision=False):
    ...   print(jnp.float64(1.0).dtype.name)
    float32
  """
  old_config = jax.config.jax_enable_x64
  try:
    jax.config.update('jax_enable_x64', enable_x64_precision)
    yield
  finally:
    jax.config.update('jax_enable_x64', old_config)


# TODO(b/183800387): remove legacy aliases.
safe_norm = numerics.safe_norm
safe_int32_increment = numerics.safe_int32_increment
