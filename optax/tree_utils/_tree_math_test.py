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

import jax.numpy as jnp
import numpy as np

from optax import tree_utils as tu


class TreeUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.tree_a = (rng.randn(20, 10) + 1j * rng.randn(20, 10), rng.randn(20))
    self.tree_b = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict = (1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    self.tree_b_dict = (1.0, {'k1': 2.0, 'k2': (3.0, 4.0)}, 5.0)

    self.array_a = rng.randn(20) + 1j * rng.randn(20)
    self.array_b = rng.randn(20)

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
    np.testing.assert_allclose(expected, got)

if __name__ == '__main__':
  absltest.main()
