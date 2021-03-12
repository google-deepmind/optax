# Lint as: python3
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

from typing import Sequence
from functools import partial

import chex
import jax
import jax.numpy as jnp

from optax._src import transform


class DifferentiallyPrivateAggregateState(transform.OptState):
  """State containing PRNGKey for differentially_private_aggregate."""
  rng_key: jnp.array


@partial(jax.vmap, in_axes=(0, None))
def _clip_per_example_grads(grads_flat: Sequence[chex.Array],
                            l2_norm_clip: float):
  global_grad_norm = jnp.linalg.norm([
      jnp.linalg.norm(g.ravel()) for g in grads_flat])
  divisor = jnp.maximum(global_grad_norm / l2_norm_clip, 1.0)
  return [g/divisor for g in grads_flat]


def differentially_private_aggregate(
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int) -> transform.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  WARNING: Unlike other transforms, `differentially_private_aggregate` expects
  the input updates to have a batch dimension in the 0th axis. That is, this
  function expects per-example gradients as input (which are easy to obtain in
  JAX using `jax.vmap`). It can still be composed with other transformations as
  long as it is the first in the chain.

  References:
    [Song et al, 2013](https://cseweb.ucsd.edu/~kamalika/pubs/scs13.pdf)

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    seed: initial seed used for the jax.random.PRNGKey

  Returns:
    A `GradientTransformation`.
  """
  noise_std = l2_norm_clip * noise_multiplier

  def init_fn(_):
    return DifferentiallyPrivateAggregateState(rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):
    grads_flat, grads_treedef = jax.tree_flatten(updates)

    if params is not None and any(g.shape[1:] != p.shape
        for g, p in zip(grads_flat, jax.tree_leaves(params))):
      raise ValueError(
          'Unlike other transforms, `differentially_private_aggregate` expects'
          ' `updates` to have a batch dimension in the 0th axis. That is, this'
          ' function expects per-example gradients as input.')

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat)+1)
    clipped_grads_flat = _clip_per_example_grads(grads_flat, l2_norm_clip)
    updates_flat = [
        (g.sum(0) + noise_std * jax.random.normal(r, g.shape[1:])) / g.shape[0]
        for r, g in zip(rngs, clipped_grads_flat)]
    return (jax.tree_unflatten(grads_treedef, updates_flat),
            DifferentiallyPrivateAggregateState(rng_key=new_key))

  return transform.GradientTransformation(init_fn, update_fn)
