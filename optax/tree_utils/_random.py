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
"""Utilities to generate random pytrees."""

from typing import Callable

import chex
import jax
from jax import tree_util as jtu

from optax._src import base


def _tree_rng_keys_split(
    rng_key: chex.PRNGKey, target_tree: chex.ArrayTree
) -> chex.ArrayTree:
  """Split keys to match structure of target tree.

  Args:
    rng_key: the key to split.
    target_tree: the tree whose structure to match.

  Returns:
    a tree of rng keys.
  """
  tree_def = jtu.tree_structure(target_tree)
  keys = jax.random.split(rng_key, tree_def.num_leaves)
  return jtu.tree_unflatten(tree_def, keys)


def tree_random_like(
    rng_key: chex.PRNGKey,
    target_tree: chex.ArrayTree,
    sampler: Callable[
        [chex.PRNGKey, base.Shape], chex.Array
    ] = jax.random.normal,
) -> chex.ArrayTree:
  """Create tree with random entries of the same shape as target tree.

  Args:
    rng_key: the key for the random number generator.
    target_tree: the tree whose structure to match. Leaves must be arrays.
    sampler: the noise sampling function, by default ``jax.random.normal``.

  Returns:
    a random tree with the same structure as ``target_tree``, whose leaves have
    distribution ``sampler``.

  .. versionadded:: 0.2.1
  """
  keys_tree = _tree_rng_keys_split(rng_key, target_tree)
  return jtu.tree_map(
      lambda l, k: sampler(k, l.shape),
      target_tree,
      keys_tree,
  )
