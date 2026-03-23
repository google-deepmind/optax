# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the MADGRAD optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _madgrad


class MadgradTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _madgrad.madgrad(learning_rate=1e-2)
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_stores_x0(self):
    """MADGRAD stores initial parameters x0 for dual averaging."""
    opt = _madgrad.scale_by_madgrad(learning_rate=1e-2)
    state = opt.init(self.params)
    jnp.testing.assert_array_equal(state.x0, self.params)

  @parameterized.product(learning_rate=(1e-4, 1e-2))
  def test_single_step_finite(self, learning_rate):
    opt = _madgrad.madgrad(learning_rate=learning_rate)
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_requires_params(self):
    opt = _madgrad.madgrad(learning_rate=1e-2)
    state = opt.init(self.params)
    with self.assertRaises(ValueError):
      opt.update(self.grads, state, params=None)

  def test_count_increments(self):
    opt = _madgrad.scale_by_madgrad(learning_rate=1e-2)
    state = opt.init(self.params)
    self.assertEqual(int(state.count), 0)
    _, state = opt.update(self.grads, state, self.params)
    self.assertEqual(int(state.count), 1)

  def test_zero_gradients(self):
    opt = _madgrad.madgrad(learning_rate=1e-2)
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  @parameterized.product(momentum=(0.0, 0.9))
  def test_momentum_values(self, momentum):
    opt = _madgrad.madgrad(learning_rate=1e-2, momentum=momentum)
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
