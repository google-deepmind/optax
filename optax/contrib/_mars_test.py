# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
    # At step 0, correction should be 0, so it should behave like Adam(correction=0)
    # We verify this by ensuring the update is just the Adam direction.
    params = {'w': jnp.zeros((1,))}
    updates = {'w': jnp.array([1.0])}
    
    opt = _mars.scale_by_mars(b1=0.9, b2=0.999, eps=0.0)
    state = opt.init(params)
    
    # Update logic:
    # m = 0.1 * 1.0 = 0.1
    # v = 0.001 * 1.0^2 = 0.001
    # m_hat = 0.1 / (1 - 0.9) = 1.0
    # v_hat = 0.001 / (1 - 0.999) = 1.0
    # adam = 1.0 / sqrt(1.0) = 1.0
    # correction should be 0 because t=0
    # final = 1.0
    
    new_updates, new_state = opt.update(updates, state, params)
    
    self.assertEqual(new_state.count, 1)
    chex.assert_trees_all_close(new_updates, {'w': jnp.array([1.0])})
    
    # Check that prev_grad is updated
    chex.assert_trees_all_close(new_state.prev_grad, updates)

  def test_max_norm_clipping(self):
    # Test that a very large gradient difference is clipped
    params = {'w': jnp.zeros((1,))}
    opt = _mars.scale_by_mars(
        b1=0.9, 
        gamma=1.0, # High gamma to amplify correction
        max_norm=0.1 # Very small clipping threshold
    )
    
    # Step 1: Initial step
    # g_0 = 1.0
    # prev_grad becomes 1.0
    updates_0 = {'w': jnp.array([1.0])}
    state = opt.init(params)
    _, state = opt.update(updates_0, state, params)
    
    # Step 2: Large jump in gradient
    # g_1 = 100.0
    # diff = 100.0 - 1.0 = 99.0
    # coeff = gamma * (b1 / 1-b1) = 1.0 * (0.9/0.1) = 9.0
    # raw_correction = 9.0 * 99.0 = 891.0
    # This should be clipped to max_norm=0.1
    updates_1 = {'w': jnp.array([100.0])}
    
    # We need to calculate what the Adam part will be roughly, 
    # but since we only care that clipping happened, we can inspect 
    # that the values are within a reasonable range or specifically test the correction logic if possible.
    # Since we can't easily isolate correction in the output without calcualting Adam,
    # let's do a more direct check or calculation.
    
    # Calculate expected Adam part
    # m = 0.9*0.1 + 0.1*100 = 0.09 + 10 = 10.09
    # v = ...
    # ...
    
    # Actually, let's just assert that the total update is not huge.
    # If correction wasn't clipped, it would be +891.0 roughly.
    # If clipped, it is +0.1 (sign preserved).
    # Adam part is roughly magnitude of 1 (standardized).
    # So result should be around 1.1, definitely not 890.
    
    new_updates, _ = opt.update(updates_1, state, params)
    
    # The Adam update part is normalized, so it's around 1.0 or -1.0.
    # Max norm is 0.1
    # Result should be small.
    self.assertLess(jnp.abs(new_updates['w'][0]), 10.0)

  def test_mars_adamw_wrapper(self):
    # Just check it runs and produces expected structure
    opt = _mars.mars_adamw(learning_rate=0.01)
    params = {'w': jnp.array([1.0])}
    state = opt.init(params)
    # count, mu, nu, prev_grad (from mars), empty_state (from decay)
    # scale_by_learning_rate has EmptyState (or ScalByScheduleState if schedule).
    # structure: (MarsState, EmptyState, EmptyState)
    
    # Structure: (MarsState, EmptyState, EmptyState)
    self.assertEqual(len(state), 3)
    self.assertIsInstance(state[0], _mars.MarsState)

if __name__ == '__main__':
  absltest.main()
