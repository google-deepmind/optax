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
"""Tests for the ACProp optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax._src import update
from optax.contrib import _acprop


class AcpropTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _acprop.acprop(learning_rate=1e-3)
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  @parameterized.product(learning_rate=(1e-4, 1e-2))
  def test_single_step_finite(self, learning_rate):
    opt = _acprop.acprop(learning_rate=learning_rate)
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))
    new_leaves = jax.tree.leaves(new_state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in new_leaves))

  def test_zero_gradients(self):
    opt = _acprop.acprop(learning_rate=1e-3)
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    test_utils.assert_trees_all_close(updates, jnp.zeros_like(self.params))

  def test_first_step_uses_raw_gradient(self):
    """On the first step, nu_hat is forced to 1, so update = grad / (1 + eps)."""
    opt = _acprop.scale_by_acprop(eps=1e-16)
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state)
    # First step: nu_hat forced to 1, so updates = grads / (sqrt(1) + eps)
    expected = self.grads / (1.0 + 1e-16)
    test_utils.assert_trees_all_close(updates, expected, atol=1e-6)

  def test_weight_decay(self):
    opt_wd = _acprop.acprop(learning_rate=1e-3, weight_decay=0.1)
    opt_nowd = _acprop.acprop(learning_rate=1e-3, weight_decay=0.0)
    state_wd = opt_wd.init(self.params)
    state_nowd = opt_nowd.init(self.params)
    u_wd, _ = opt_wd.update(self.grads, state_wd, self.params)
    u_nowd, _ = opt_nowd.update(self.grads, state_nowd, self.params)
    # Weight decay should make the updates differ.
    self.assertFalse(jnp.allclose(u_wd, u_nowd))


if __name__ == '__main__':
  absltest.main()
