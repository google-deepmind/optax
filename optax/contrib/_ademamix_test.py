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
"""Tests for the AdEMAMix optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax._src import update
from optax.contrib import _ademamix


class AdemamixTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _ademamix.ademamix(learning_rate=1e-3)
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_init_has_dual_momentum(self):
    """AdEMAMix should have both fast (m1) and slow (m2) EMA buffers."""
    opt = _ademamix.scale_by_ademamix()
    state = opt.init(self.params)
    self.assertEqual(state.m1.shape, self.params.shape)
    self.assertEqual(state.m2.shape, self.params.shape)
    self.assertEqual(state.nu.shape, self.params.shape)

  @parameterized.product(learning_rate=(1e-4, 1e-2))
  def test_single_step_finite(self, learning_rate):
    opt = _ademamix.ademamix(learning_rate=learning_rate)
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_zero_gradients(self):
    opt = _ademamix.ademamix(learning_rate=1e-3)
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    test_utils.assert_trees_all_close(updates, jnp.zeros_like(self.params))

  def test_callable_b3_and_alpha(self):
    """b3 and alpha can be schedules (callables)."""
    b3_schedule = lambda t: 0.999 + 0.0001 * t / (t + 1)
    alpha_schedule = lambda t: 5.0 + t * 0.01
    opt = _ademamix.ademamix(
        learning_rate=1e-3, b3=b3_schedule, alpha=alpha_schedule
    )
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_simplified_ademamix_single_step(self):
    opt = _ademamix.simplified_ademamix(learning_rate=1e-3)
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
