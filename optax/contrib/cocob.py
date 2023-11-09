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
"""Backpropagating variant of the COntinuous COin Betting stochastic algorithm.

COCOB is a contributed optimizer implemented from Algorithm 2 of "Training Deep
Networks without Learning Rates Through Coin Betting" by Francesco Orabona and
Tatiana Tommasi.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax.tree_util import tree_map

from optax._src import base


class COCOBState(NamedTuple):
  """State for COntinuous COin Betting."""
  init_particles: base.Updates
  cumulative_gradients: base.Updates
  scale: base.Updates
  subgradients: base.Updates
  reward: base.Updates


def cocob(
    alpha: float = 100, eps: float = 1e-8
) -> base.GradientTransformation:
  """Rescale updates according to the COntinuous COin Betting algorithm.

  Algorithm for stochastic subgradient descent. Uses a gambling algorithm to 
  find the minimizer of a non-smooth objective function by accessing its 
  subgradients. All we need is a good gambling strategy. See Algorithm 2 of:

  References:
    [Orabona & Tommasi, 2017](https://proceedings.neurips.cc/paper/2017/file/7c82fab8c8f89124e2ce92984e04fb40-Paper.pdf) #pylint:disable=line-too-long

  Args:
    alpha: fraction to bet parameter of the COCOB optimizer
    eps: jitter term to avoid dividing by 0

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    init_adapt = tree_map(lambda p: jnp.zeros(p.shape), params)
    init_scale = tree_map(lambda p: eps * jnp.ones(p.shape), params)
    return COCOBState(init_particles=params, cumulative_gradients=init_adapt,
                      scale=init_scale, subgradients=init_adapt,
                      reward=init_adapt)

  def update_fn(updates, state, params):
    init_particles, cumulative_grads, scale, subgradients, reward = state

    scale = tree_map(lambda L, c: jnp.maximum(L, jnp.abs(c)), scale, updates)
    subgradients = tree_map(lambda G, c: G + jnp.abs(c), subgradients, updates)
    reward = tree_map(
        lambda R, c, p, p0: jnp.maximum(R - c * (p - p0), 0),
        reward, updates, params, init_particles)
    cumulative_grads = tree_map(lambda C, c: C - c, cumulative_grads, updates)

    new_updates = tree_map(lambda p, p0, C, L, G, R: (
        -p + (p0 + C / (L * jnp.maximum(G + L, alpha * L)) * (L + R))),
                           params, init_particles, cumulative_grads, scale,
                           subgradients, reward)

    new_state = COCOBState(init_particles=init_particles,
                           cumulative_gradients=cumulative_grads, scale=scale,
                           subgradients=subgradients, reward=reward)
    return new_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)
