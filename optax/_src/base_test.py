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
"""Tests for base.py."""

from absl.testing import absltest
import chex
import numpy as np

from optax._src import base

# pylint:disable=no-value-for-parameter


class BaseTest(chex.TestCase):

  def test_typing(self):
    """Ensure that the type annotations work for the update function."""

    def f(updates, opt_state, params=None):
      del params
      return updates, opt_state

    def g(f: base.TransformUpdateFn):
      updates = np.zeros([])
      params = np.zeros([])
      opt_state = np.zeros([])

      f(updates, opt_state)
      f(updates, opt_state, params)
      f(updates, opt_state, params=params)

    g(f)

  @chex.all_variants
  def test_set_to_zero_returns_tree_of_correct_zero_arrays(self):
    """Tests that zero transform returns a tree of zeros of correct shape."""
    grads = ({'a': np.ones((3, 4)), 'b': 1.}, np.ones((1, 2, 3)))
    updates, _ = self.variant(base.set_to_zero().update)(grads,
                                                         base.EmptyState())
    correct_zeros = ({'a': np.zeros((3, 4)), 'b': 0.}, np.zeros((1, 2, 3)))
    chex.assert_trees_all_close(updates, correct_zeros, rtol=0)

  @chex.all_variants(with_pmap=False)
  def test_set_to_zero_is_stateless(self):
    """Tests that the zero transform returns an empty state."""
    self.assertEqual(
        self.variant(base.set_to_zero().init)(params=None), base.EmptyState())


if __name__ == '__main__':
  absltest.main()
