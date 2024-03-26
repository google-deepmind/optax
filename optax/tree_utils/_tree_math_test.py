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

"""Tests for optax.tree_utils."""

from absl.testing import absltest
import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np
from optax import tree_utils as tu


class TreeUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.rng_jax = jax.random.PRNGKey(0)

    self.tree_a = (rng.randn(20, 10) + 1j * rng.randn(20, 10), rng.randn(20))
    self.tree_b = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict = (1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    self.tree_b_dict = (1.0, {'k1': 2.0, 'k2': (3.0, 4.0)}, 5.0)

    self.array_a = rng.randn(20) + 1j * rng.randn(20)
    self.array_b = rng.randn(20)

    self.tree_a_dict_jax = jtu.tree_map(jnp.array, self.tree_a_dict)
    self.tree_b_dict_jax = jtu.tree_map(jnp.array, self.tree_b_dict)

  def test_tree_add(self):
    expected = self.array_a + self.array_b
    got = tu.tree_add(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (self.tree_a[0] + self.tree_b[0],
                self.tree_a[1] + self.tree_b[1])
    got = tu.tree_add(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_sub(self):
    expected = self.array_a - self.array_b
    got = tu.tree_sub(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (self.tree_a[0] - self.tree_b[0],
                self.tree_a[1] - self.tree_b[1])
    got = tu.tree_sub(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_mul(self):
    expected = self.array_a * self.array_b
    got = tu.tree_mul(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (self.tree_a[0] * self.tree_b[0],
                self.tree_a[1] * self.tree_b[1])
    got = tu.tree_mul(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_div(self):
    expected = self.array_a / self.array_b
    got = tu.tree_div(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (self.tree_a[0] / self.tree_b[0],
                self.tree_a[1] / self.tree_b[1])
    got = tu.tree_div(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_scalar_mul(self):
    expected = 0.5 * self.array_a
    got = tu.tree_scalar_mul(0.5, self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (0.5 * self.tree_a[0], 0.5 * self.tree_a[1])
    got = tu.tree_scalar_mul(0.5, self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_tree_add_scalar_mul(self):
    expected = (self.tree_a[0] + 0.5 * self.tree_b[0],
                self.tree_a[1] + 0.5 * self.tree_b[1])
    got = tu.tree_add_scalar_mul(self.tree_a, 0.5, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_vdot(self):
    expected = jnp.vdot(self.array_a, self.array_b)
    got = tu.tree_vdot(self.array_a, self.array_b)
    np.testing.assert_allclose(expected, got)

    expected = 15.0
    got = tu.tree_vdot(self.tree_a_dict, self.tree_b_dict)
    np.testing.assert_allclose(expected, got)

    expected = (jnp.vdot(self.tree_a[0], self.tree_b[0]) +
                jnp.vdot(self.tree_a[1], self.tree_b[1]))
    got = tu.tree_vdot(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_sum(self):
    expected = jnp.sum(self.array_a)
    got = tu.tree_sum(self.array_a)
    np.testing.assert_allclose(expected, got)

    expected = (jnp.sum(self.tree_a[0]) + jnp.sum(self.tree_a[1]))
    got = tu.tree_sum(self.tree_a)
    np.testing.assert_allclose(expected, got)

  def test_tree_l2_norm(self):
    expected = jnp.sqrt(jnp.vdot(self.array_a, self.array_a).real)
    got = tu.tree_l2_norm(self.array_a)
    np.testing.assert_allclose(expected, got)

    expected = jnp.sqrt(jnp.vdot(self.tree_a[0], self.tree_a[0]).real +
                        jnp.vdot(self.tree_a[1], self.tree_a[1]).real)
    got = tu.tree_l2_norm(self.tree_a)
    np.testing.assert_allclose(expected, got)

  def test_tree_zeros_like(self):
    expected = jnp.zeros_like(self.array_a)
    got = tu.tree_zeros_like(self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (jnp.zeros_like(self.tree_a[0]), jnp.zeros_like(self.tree_a[1]))
    got = tu.tree_zeros_like(self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_tree_ones_like(self):
    expected = jnp.ones_like(self.array_a)
    got = tu.tree_ones_like(self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (jnp.ones_like(self.tree_a[0]), jnp.ones_like(self.tree_a[1]))
    got = tu.tree_ones_like(self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_add_multiple_trees(self):
    """Test adding more than 2 trees with tree_add."""
    trees = [self.tree_a_dict_jax, self.tree_a_dict_jax, self.tree_a_dict_jax]
    expected = tu.tree_scalar_mul(3., self.tree_a_dict_jax)
    got = tu.tree_add(*trees)
    chex.assert_trees_all_close(expected, got)

  def test_random_like_tree(self, eps=1e-6):
    """Test for `tree_random_like`.

    Args:
      eps: amount of noise.

    Tests that `tree_random_like` generates a tree of the proper structure,
    that it can be added to a target tree with a small multiplicative factor
    without errors, and that the resulting addition is close to the original.
    """
    rand_tree_a = tu.tree_random_like(self.rng_jax, self.tree_a)
    rand_tree_b = tu.tree_random_like(self.rng_jax, self.tree_b)
    rand_tree_a_dict = tu.tree_random_like(self.rng_jax, self.tree_a_dict_jax)
    rand_tree_b_dict = tu.tree_random_like(self.rng_jax, self.tree_b_dict_jax)
    rand_array_a = tu.tree_random_like(self.rng_jax, self.array_a)
    rand_array_b = tu.tree_random_like(self.rng_jax, self.array_b)
    sum_tree_a = tu.tree_add_scalar_mul(self.tree_a, eps, rand_tree_a)
    sum_tree_b = tu.tree_add_scalar_mul(self.tree_b, eps, rand_tree_b)
    sum_tree_a_dict = tu.tree_add_scalar_mul(self.tree_a_dict,
                                             eps,
                                             rand_tree_a_dict)
    sum_tree_b_dict = tu.tree_add_scalar_mul(self.tree_b_dict,
                                             eps,
                                             rand_tree_b_dict)
    sum_array_a = tu.tree_add_scalar_mul(self.array_a, eps, rand_array_a)
    sum_array_b = tu.tree_add_scalar_mul(self.array_b, eps, rand_array_b)
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
