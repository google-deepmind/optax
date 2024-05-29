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
"""Tests for optax.tree_utils._random."""

from absl.testing import absltest
import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np

from optax import tree_utils as otu


class RandomTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.rng_jax = jax.random.PRNGKey(0)

    self.tree_a = (rng.randn(20, 10) + 1j * rng.randn(20, 10), rng.randn(20))
    self.tree_b = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict = jtu.tree_map(
        jnp.asarray,
        (
            1.0,
            {'k1': 1.0, 'k2': (1.0, 1.0)},
            1.0
        )
    )
    self.tree_b_dict = jtu.tree_map(
        jnp.asarray,
        (
            1.0,
            {'k1': 2.0, 'k2': (3.0, 4.0)},
            5.0
        )
    )

    self.array_a = rng.randn(20) + 1j * rng.randn(20)
    self.array_b = rng.randn(20)

    self.tree_a_dict_jax = jtu.tree_map(jnp.array, self.tree_a_dict)
    self.tree_b_dict_jax = jtu.tree_map(jnp.array, self.tree_b_dict)

  def test_tree_random_like(self, eps=1e-6):
    """Test for `tree_random_like`.

    Args:
      eps: amount of noise.

    Tests that `tree_random_like` generates a tree of the proper structure,
    that it can be added to a target tree with a small multiplicative factor
    without errors, and that the resulting addition is close to the original.
    """
    rand_tree_a = otu.tree_random_like(self.rng_jax, self.tree_a)
    rand_tree_b = otu.tree_random_like(self.rng_jax, self.tree_b)
    rand_tree_a_dict = otu.tree_random_like(self.rng_jax, self.tree_a_dict_jax)
    rand_tree_b_dict = otu.tree_random_like(self.rng_jax, self.tree_b_dict_jax)
    rand_array_a = otu.tree_random_like(self.rng_jax, self.array_a)
    rand_array_b = otu.tree_random_like(self.rng_jax, self.array_b)
    sum_tree_a = otu.tree_add_scalar_mul(self.tree_a, eps, rand_tree_a)
    sum_tree_b = otu.tree_add_scalar_mul(self.tree_b, eps, rand_tree_b)
    sum_tree_a_dict = otu.tree_add_scalar_mul(self.tree_a_dict,
                                              eps,
                                              rand_tree_a_dict)
    sum_tree_b_dict = otu.tree_add_scalar_mul(self.tree_b_dict,
                                              eps,
                                              rand_tree_b_dict)
    sum_array_a = otu.tree_add_scalar_mul(self.array_a, eps, rand_array_a)
    sum_array_b = otu.tree_add_scalar_mul(self.array_b, eps, rand_array_b)
    tree_sums = [sum_tree_a,
                 sum_tree_b,
                 sum_tree_a_dict,
                 sum_tree_b_dict,
                 sum_array_a,
                 sum_array_b]
    trees = [self.tree_a,
             self.tree_b,
             self.tree_a_dict,
             self.tree_b_dict,
             self.array_a,
             self.array_b]
    chex.assert_trees_all_close(trees, tree_sums, atol=1e-5)

if __name__ == '__main__':
  absltest.main()
