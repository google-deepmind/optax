# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Differential Privacy utilities."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import clipping


# pylint:disable=no-value-for-parameter
class DifferentiallyPrivateAggregateState(NamedTuple):
  """State containing PRNGKey for `differentially_private_aggregate`."""
  rng_key: jnp.array


def differentially_private_aggregate(
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int
) -> base.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  WARNING: Unlike other transforms, `differentially_private_aggregate` expects
  the input updates to have a batch dimension in the 0th axis. That is, this
  function expects per-example gradients as input (which are easy to obtain in
  JAX using `jax.vmap`). It can still be composed with other transformations as
  long as it is the first in the chain.

  References:
    [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    seed: initial seed used for the jax.random.PRNGKey

  Returns:
    A `GradientTransformation`.
  """
  noise_std = l2_norm_clip * noise_multiplier

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree_util.tree_flatten(updates)
    bsize = grads_flat[0].shape[0]
    clipped, _ = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [(g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
              for g, r in zip(clipped, rngs)]
    return (jax.tree_util.tree_unflatten(grads_treedef, noised),
            DifferentiallyPrivateAggregateState(rng_key=new_key))

  return base.GradientTransformation(init_fn, update_fn)
