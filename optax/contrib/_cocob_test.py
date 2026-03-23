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
"""Tests for the COCOB optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _cocob


class CocobTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _cocob.cocob()
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_stores_init_particles(self):
    """COCOB stores initial params (init_particles) in state."""
    opt = _cocob.scale_by_cocob()
    state = opt.init(self.params)
    jnp.testing.assert_array_equal(state.init_particles, self.params)

  def test_single_step_finite(self):
    opt = _cocob.cocob()
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))
    new_leaves = jax.tree.leaves(new_state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in new_leaves))

  def test_zero_gradients(self):
    opt = _cocob.cocob()
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  @parameterized.product(weight_decay=(0.0, 0.01))
  def test_weight_decay(self, weight_decay):
    opt = _cocob.cocob(weight_decay=weight_decay)
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_scale_tracks_max_gradient(self):
    """The scale field should track the element-wise max abs gradient."""
    opt = _cocob.scale_by_cocob()
    state = opt.init(self.params)
    grads_small = jnp.array([0.01, 0.02, 0.03])
    _, state = opt.update(grads_small, state, self.params)
    grads_big = jnp.array([1.0, 2.0, 3.0])
    _, state = opt.update(grads_big, state, self.params)
    # Scale should be at least as large as the big gradients.
    self.assertTrue(jnp.all(state.scale >= jnp.abs(grads_big)))


if __name__ == '__main__':
  absltest.main()
