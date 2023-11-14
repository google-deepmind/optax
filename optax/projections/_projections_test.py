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
    # test with array
    x = jnp.array([-1.0, 2.0, 3.0])
    expected = jnp.array([0, 2.0, 3.0])
    np.testing.assert_array_equal(proj.projection_non_negative(x), expected)

    # test with tuple
    np.testing.assert_array_equal(proj.projection_non_negative((x, x)),
                                  (expected, expected))

    # test with nested pytree
    tree_x = (-1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    tree_expected = (0.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    chex.assert_trees_all_equal(proj.projection_non_negative(tree_x),
                                tree_expected)

if __name__ == '__main__':
  absltest.main()
