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
"""Gradient clipping transformations."""

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import linear_algebra

# pylint:disable=no-value-for-parameter


ClipState = base.EmptyState


def clip(max_delta) -> base.GradientTransformation:
  """Clip updates element-wise, to be between -max_delta and +max_delta.

  Args:
    max_delta: the maximum absolute value for each element in the update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(
        lambda g: jnp.clip(g, -max_delta, max_delta), updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def clip_by_block_rms(threshold: float) -> base.GradientTransformation:
  """Clip updates to a max rms for the gradient of each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    threshold: the maximum rms for the gradient of each param vector or matrix.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return base.EmptyState()

  def update_fn(updates, state, params=None):
    del params
    def _clip_fn(u):
      clip_denom = (jnp.maximum(1.0, jnp.sqrt(jnp.mean(u ** 2)) / threshold))
      return u / clip_denom
    updates = jax.tree_map(_clip_fn, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


ClipByGlobalNormState = base.EmptyState


def clip_by_global_norm(max_norm) -> base.GradientTransformation:
  """Clip updates using their global norm.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    max_norm: the maximum global norm for an update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipByGlobalNormState()

  def update_fn(updates, state, params=None):
    del params
    g_norm = linear_algebra.global_norm(updates)
    # TODO(b/163995078): revert back to the following (faster) implementation
    # once analysed how it affects backprop through update (e.g. meta-gradients)
    # g_norm = jnp.maximum(max_norm, g_norm)
    # updates = jax.tree_map(lambda t: (t / g_norm) * max_norm, updates)
    trigger = g_norm < max_norm
    updates = jax.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def unitwise_norm(x):
  """Computes norms of each output unit separately."""
  if len(jnp.squeeze(x).shape) <= 1:  # Scalars and vectors
    axis = None
    keepdims = False
  # Note that this assumes parameters with a shape of length 3 are multihead
  # linear parameters--if you wish to apply AGC to 1D convs, you may need
  # to modify this line.
  elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
    axis = 0
    keepdims = True
  elif len(x.shape) == 4:  # Conv kernels of shape HWIO
    axis = [0, 1, 2,]
    keepdims = True
  else:
    raise ValueError(f'Got a parameter with shape not in [1, 2, 3, 4]! {x}')
  return jnp.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_clip(g_norm, max_norm, grad):
  """Applies gradient clipping unit-wise."""
  trigger = g_norm < max_norm
  # This little max(., 1e-6) is distinct from the normal eps and just prevents
  # division by zero. It technically should be impossible to engage.
  clipped_grad = grad * (max_norm / jnp.maximum(g_norm, 1e-6))
  return jnp.where(trigger, grad, clipped_grad)


AdaptiveGradClipState = base.EmptyState


def adaptive_grad_clip(clipping, eps=1e-3) -> base.GradientTransformation:
  """Clip updates to be at most clipping * parameter_norm, unit-wise.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

  Args:
    clipping: Maximum allowed ratio of update norm to parameter norm.
    eps: epsilon term to prevent clipping of zero-initialized params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AdaptiveGradClipState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    g_norm = jax.tree_map(unitwise_norm, updates)
    p_norm = jax.tree_map(unitwise_norm, params)
    # Maximum allowable norm
    max_norm = jax.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
    # If grad norm > clipping * param_norm, rescale
    updates = jax.tree_multimap(unitwise_clip, g_norm, max_norm, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)
