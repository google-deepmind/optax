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
"""Gradient clipping transformations.

Note that complex numbers are also supported, see
https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
"""

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import linear_algebra
from optax._src import numerics

ClipState = base.EmptyState


def clip(max_delta: chex.Numeric) -> base.GradientTransformation:
  """Clips updates element-wise, to be in ``[-max_delta, +max_delta]``.

  Args:
    max_delta: The maximum absolute value for each element in the update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return ClipState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(lambda g: jnp.clip(g, -max_delta, max_delta),
                           updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def clip_by_block_rms(threshold: float) -> base.GradientTransformation:
  """Clips updates to a max rms for the gradient of each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    threshold: The maximum rms for the gradient of each param vector or matrix.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return base.EmptyState()

  def update_fn(updates, state, params=None):
    del params

    def _clip_fn(u):
      clip_denom = jnp.maximum(
          1.0,
          jnp.sqrt(jnp.mean(numerics.abs_sq(u))) / threshold)
      return u / clip_denom

    updates = jax.tree_map(_clip_fn, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


ClipByGlobalNormState = base.EmptyState


def clip_by_global_norm(max_norm: float) -> base.GradientTransformation:
  """Clips updates using their global norm.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    max_norm: The maximum global norm for an update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return ClipByGlobalNormState()

  def update_fn(updates, state, params=None):
    del params
    g_norm = linear_algebra.global_norm(updates)
    # TODO(b/163995078): revert back to the following (faster) implementation
    # once analysed how it affects backprop through update (e.g. meta-gradients)
    # g_norm = jnp.maximum(max_norm, g_norm)
    # updates = jax.tree_map(lambda t: (t / g_norm) * max_norm, updates)
    trigger = jnp.squeeze(g_norm < max_norm)
    chex.assert_shape(trigger, ())  # A scalar.

    updates = jax.tree_map(
        lambda t: jax.lax.select(trigger, t, (t / g_norm) * max_norm), updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def unitwise_norm(x: chex.Array) -> chex.Array:
  """Computes norms of each output unit separately."""
  if jnp.squeeze(x).ndim <= 1:  # Scalars and vectors
    squared_norm = jnp.sum(numerics.abs_sq(x), keepdims=True)
  # Note that this assumes parameters with a shape of length 3 are multihead
  # linear parameters--if you wish to apply AGC to 1D convs, you may need
  # to modify this line.
  elif x.ndim in (2, 3):  # Linear layers of shape IO or multihead linear
    squared_norm = jnp.sum(numerics.abs_sq(x), axis=0, keepdims=True)
  elif x.ndim == 4:  # Conv kernels of shape HWIO
    squared_norm = jnp.sum(numerics.abs_sq(x), axis=(0, 1, 2), keepdims=True)
  else:
    raise ValueError(
        f'Expected parameter with shape in {1, 2, 3, 4}, got {x.shape}.')
  chex.assert_is_broadcastable(squared_norm.shape, x.shape)
  return jnp.broadcast_to(jnp.sqrt(squared_norm), x.shape)


def unitwise_clip(g_norm: chex.Array,
                  max_norm: chex.Array,
                  grad: chex.Array,
                  div_eps: float = 1e-6) -> chex.Array:
  """Applies gradient clipping unit-wise."""
  # This little max(., div_eps) is distinct from the normal eps and just
  # prevents division by zero. It technically should be impossible to engage.
  clipped_grad = grad * (max_norm / jnp.maximum(g_norm, div_eps))
  chex.assert_equal_shape((g_norm, max_norm, grad, clipped_grad))
  return jnp.where(g_norm < max_norm, grad, clipped_grad)


AdaptiveGradClipState = base.EmptyState


def adaptive_grad_clip(clipping: float,
                       eps: float = 1e-3) -> base.GradientTransformation:
  """Clips updates to be at most ``clipping * parameter_norm``, unit-wise.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

  Args:
    clipping: The maximum allowed ratio of update norm to parameter norm.
    eps: An epsilon term to prevent clipping of zero-initialized params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return AdaptiveGradClipState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    g_norm, p_norm = jax.tree_map(unitwise_norm, (updates, params))
    # Maximum allowable norm.
    max_norm = jax.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
    # If grad norm > clipping * param_norm, rescale.
    updates = jax.tree_map(unitwise_clip, g_norm, max_norm, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)
