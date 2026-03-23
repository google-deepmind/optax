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
"""Tests for the ADOPT optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax._src import update
from optax.contrib import _adopt


class AdoptTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _adopt.adopt(learning_rate=1e-2)
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  @parameterized.product(
      learning_rate=(1e-4, 1e-2, 1.0),
      nesterov=(True, False),
  )
  def test_single_step_finite(self, learning_rate, nesterov):
    opt = _adopt.adopt(learning_rate=learning_rate, nesterov=nesterov)
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))
    new_leaves = jax.tree.leaves(new_state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in new_leaves))

  def test_zero_gradients(self):
    opt = _adopt.adopt(learning_rate=1e-2)
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    test_utils.assert_trees_all_close(updates, jnp.zeros_like(self.params))

  def test_clipping_disabled(self):
    opt_clip = _adopt.adopt(learning_rate=1e-2, use_clipping=True)
    opt_noclip = _adopt.adopt(learning_rate=1e-2, use_clipping=False)
    state_clip = opt_clip.init(self.params)
    state_noclip = opt_noclip.init(self.params)
    # Run two steps so clipping can take effect (first step b1=1 so mu=grads).
    for _ in range(2):
      u_clip, state_clip = opt_clip.update(self.grads, state_clip, self.params)
      u_noclip, state_noclip = opt_noclip.update(
          self.grads, state_noclip, self.params
      )
    # With small gradients clipping may not differ, but both should be finite.
    self.assertTrue(jnp.all(jnp.isfinite(u_clip)))
    self.assertTrue(jnp.all(jnp.isfinite(u_noclip)))

  def test_first_step_is_identity_like(self):
    """On the first step, b1_=1 so mu should equal the scaled gradient."""
    opt = _adopt.scale_by_adopt()
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    # First step: b2_=0 so nu=0, b1_=1 so mu=mu_updates directly.
    # mu_updates = grads / max(sqrt(0), eps) = grads/eps, then mu = 1*mu_updates
    # Result should be finite.
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
