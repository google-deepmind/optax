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

"""Tests for optax.projections."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import numpy as np

from optax import projections as proj


class ProjectionsTest(parameterized.TestCase):

  def test_projection_non_negative(self):
    with self.subTest('with an array'):
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 3.0])
      np.testing.assert_array_equal(proj.projection_non_negative(x), expected)

    with self.subTest('with a tuple'):
      np.testing.assert_array_equal(proj.projection_non_negative((x, x)),
                                    (expected, expected))

    with self.subTest('with nested pytree'):
      tree_x = (-1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      tree_expected = (0.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      chex.assert_trees_all_equal(proj.projection_non_negative(tree_x),
                                  tree_expected)

  def test_projection_box(self):
    with self.subTest('lower and upper are scalars'):
      lower, upper = 0.0, 2.0
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 2.0])
      np.testing.assert_array_equal(proj.projection_box(x, lower, upper),
                                    expected)

    with self.subTest('lower and upper values are arrays'):
      lower_arr = jnp.ones(len(x)) * lower
      upper_arr = jnp.ones(len(x)) * upper
      np.testing.assert_array_equal(proj.projection_box(x,
                                                        lower_arr,
                                                        upper_arr),
                                    expected)

    with self.subTest('lower and upper are tuples of arrays'):
      lower_tuple = (lower, lower)
      upper_tuple = (upper, upper)
      chex.assert_trees_all_equal(proj.projection_box((x, x),
                                                      lower_tuple,
                                                      upper_tuple),
                                  (expected, expected))

    with self.subTest('lower and upper are pytrees'):
      tree = (-1.0, {'k1': 2.0, 'k2': (2.0, 3.0)}, 3.0)
      expected = (0.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      lower_tree = (0.0, {'k1': 0.0, 'k2': (0.0, 0.0)}, 0.0)
      upper_tree = (2.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      chex.assert_trees_all_equal(proj.projection_box(tree,
                                                      lower_tree,
                                                      upper_tree),
                                  expected)

  def test_projection_hypercube(self):
    x = jnp.array([-1.0, 2.0, 0.5])

    with self.subTest('with default scale'):
      expected = jnp.array([0, 1.0, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x), expected)

    with self.subTest('with scalar scale'):
      expected = jnp.array([0, 0.8, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x, 0.8), expected)

    with self.subTest('with array scales'):
      scales = jnp.ones(len(x)) * 0.8
      np.testing.assert_array_equal(proj.projection_hypercube(x, scales),
                                    expected)


if __name__ == '__main__':
  absltest.main()
