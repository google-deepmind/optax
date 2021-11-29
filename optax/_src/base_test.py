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
from absl.testing import parameterized

import chex
import numpy as np
import jax
import jax.numpy as jnp

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


class StatelessTest(chex.TestCase):
  """Tests for the stateless transformation."""

  @chex.all_variants
  @parameterized.parameters(False, True)
  def test_stateless(self, on_leaves):
    params = {'a': jnp.zeros((1, 2)), 'b': jnp.ones((1,))}
    updates = {'a': jnp.ones((1, 2)), 'b': jnp.full((1,), 2.0)}

    if on_leaves:
      def weight_decay(g, p):
        return g + 0.1 * p
    else:
      def weight_decay(g, p):
        return jax.tree_map(lambda g_, p_: g_ + 0.1 * p_, g, p)

    opt = base.stateless(weight_decay, on_leaves=on_leaves)
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    new_updates, new_state = update_fn(updates, state, params)
    expected_updates = {'a': jnp.ones((1, 2)), 'b': jnp.array([2.1])}
    chex.assert_trees_all_close(new_updates, expected_updates)

      # Check repeated application
    new_updates, _ = update_fn(new_updates, new_state, params)
    expected_updates = {'a': jnp.ones((1, 2)), 'b': jnp.array([2.2])}
    chex.assert_trees_all_close(new_updates, expected_updates)

  @chex.all_variants
  def test_stateless_no_params(self):
    updates = {'linear': jnp.full((5, 3), 3.0)}
    opt = base.stateless(lambda x, _: x*2.0, on_leaves=True)
    state = opt.init(None)
    update_fn = self.variant(opt.update)
    new_updates, _ = update_fn(updates, state)
    expected_updates = {'linear': jnp.full((5, 3), 6.0)}
    chex.assert_trees_all_close(new_updates, expected_updates)


if __name__ == '__main__':
  absltest.main()
