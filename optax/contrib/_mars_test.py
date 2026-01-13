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
"""Tests for optax.contrib._mars."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import optax
from optax.contrib import _mars


class MarsTest(parameterized.TestCase):

  def test_initialization(self):
    params = {'a': jnp.zeros((2, 3)), 'b': jnp.zeros((4,))}
    opt = _mars.scale_by_mars()
    state = opt.init(params)

    self.assertEqual(state.count, 0)
    chex.assert_trees_all_equal(state.mu, params)
    chex.assert_trees_all_equal(state.nu, params)
    chex.assert_trees_all_equal(state.prev_grad, params)

  def test_step_zero_correction(self):
    # At step 0, correction should be 0.
    params = {'w': jnp.zeros((1,))}
    updates = {'w': jnp.array([1.0])}

    opt = _mars.scale_by_mars(b1=0.9, b2=0.999, eps=0.0)
    state = opt.init(params)

    new_updates, new_state = opt.update(updates, state, params)

    self.assertEqual(new_state.count, 1)
    chex.assert_trees_all_close(new_updates, {'w': jnp.array([1.0])})
    chex.assert_trees_all_close(new_state.prev_grad, updates)

  def test_max_norm_clipping(self):
    # Test that a very large gradient difference is clipped
    params = {'w': jnp.zeros((1,))}
    opt = _mars.scale_by_mars(
        b1=0.9,
        gamma=1.0,  # High gamma to amplify correction
        max_norm=0.1  # Very small clipping threshold
    )

    # Step 1: Initial step
    updates_0 = {'w': jnp.array([1.0])}
    state = opt.init(params)
    _, state = opt.update(updates_0, state, params)

    # Step 2: Large jump in gradient.
    # diff = 100 - 1 = 99.
    # coeff = 1.0 * (0.9/0.1) = 9.0.
    # raw_correction = 9.0 * 99.0 = 891.0.
    # Clipped correction magnitude = 0.1.
    updates_1 = {'w': jnp.array([100.0])}
    new_updates, _ = opt.update(updates_1, state, params)

    # We expect the result to be roughly Adam + 0.1
    # Adam part is roughly 1.0 (since normalized).
    # So result should be around 1.1, definitely not hundreds.
    self.assertLess(jnp.abs(new_updates['w'][0]), 10.0)

  def test_gamma_zero_recovers_adam(self):
    # If gamma=0, MARS reduces to Adam.
    # Note: scale_by_mars is just the transformation.
    # We compare scale_by_mars(gamma=0) with scale_by_adam().
    # Optax adam has bias correction by default.
    params = {'w': jnp.zeros((1,))}
    updates = {'w': jnp.array([0.5])}

    mars_opt = _mars.scale_by_mars(gamma=0.0, eps=1e-8)
    adam_opt = optax.scale_by_adam(eps=1e-8)

    mars_state = mars_opt.init(params)
    adam_state = adam_opt.init(params)

    mars_updates, _ = mars_opt.update(updates, mars_state, params)
    adam_updates, _ = adam_opt.update(updates, adam_state, params)

    chex.assert_trees_all_close(mars_updates, adam_updates)

  def test_optimization(self):
    # Simple convergence test on a quadratic.
    # Loss = (x - 1.0)^2
    # Grad = 2(x - 1.0)
    start_params = jnp.array([10.0])
    opt = _mars.mars_adamw(learning_rate=0.1, gamma=0.025)
    state = opt.init(start_params)
    params = start_params

    # Run a few steps
    for _ in range(50):
      grads = 2.0 * (params - 1.0)
      updates, state = opt.update(grads, state, params)
      params = optax.apply_updates(params, updates)

    # Should be close to 1.0
    self.assertLess(jnp.abs(params[0] - 1.0), 0.1)

  def test_structure(self):
    opt = _mars.mars_adamw(learning_rate=0.01)
    params = {'w': jnp.array([1.0])}
    state = opt.init(params)
    self.assertEqual(len(state), 3)
    self.assertIsInstance(state[0], _mars.MarsState)


if __name__ == '__main__':
  absltest.main()
