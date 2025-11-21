# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for RLion optimizer."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
import numpy as np
from optax._src import update
from optax.contrib import _RLion


class RLionTest(chex.TestCase):

    def setUp(self):
        super().setUp()
        self.grads = {'x': np.array(2.0), 'y': np.array(-2.0)}
        self.initial_params = {'x': np.array(3.0), 'y': np.array(-3.0)}

    def loop(self, optimizer, num_steps, params):
        """Performs a given number of optimizer steps."""
        init_fn, update_fn = optimizer
        step = self.variant(update_fn)
        opt_state = self.variant(init_fn)(params)

        for _ in range(num_steps):
            updates, opt_state = step(self.grads, opt_state, params)
            params = update.apply_updates(params, updates)

        return params, opt_state

    @chex.all_variants(with_pmap=False)
    def test_rlion_smooth(self):
        """Test RLion with smooth sign."""
        params = self.initial_params
        optimizer = _RLion.rlion(
            learning_rate=0.1,
            b1=0.9,
            b2=0.99,
            use_smooth_sign=True
        )

        final_params, final_state = self.loop(
            optimizer=optimizer,
            num_steps=3,
            params=params
        )

        self.assertEqual(final_state[0].count, 3)
        chex.assert_tree_all_finite((final_params, final_state))

    @chex.all_variants(with_pmap=False)
    def test_rlion_hard_sign(self):
        """Test RLion with hard sign (standard Lion behavior)."""
        params = self.initial_params
        optimizer = _RLion.rlion(
            learning_rate=0.1,
            b1=0.9,
            b2=0.99,
            use_smooth_sign=False
        )

        final_params, final_state = self.loop(
            optimizer=optimizer,
            num_steps=3,
            params=params
        )

        self.assertEqual(final_state[0].count, 3)
        chex.assert_tree_all_finite((final_params, final_state))

    @chex.all_variants(with_pmap=False)
    def test_scale_by_rlion(self):
        """Test scale_by_rlion transformation."""
        params = self.initial_params
        optimizer = _RLion.scale_by_rlion(
            b1=0.9,
            b2=0.99,
            use_smooth_sign=True
        )

        final_params, final_state = self.loop(
            optimizer=optimizer,
            num_steps=1,
            params=params
        )

        # Check momentum is updated
        expected_mu_x = (1 - 0.99) * 2.0
        expected_mu_y = (1 - 0.99) * (-2.0)

        chex.assert_trees_all_close(
            final_state.mu,
            {'x': expected_mu_x, 'y': expected_mu_y},
            atol=1e-6,
        )
        chex.assert_tree_all_finite((final_params, final_state))

    def test_smooth_sign_function(self):
        """Test smooth_sign function."""
        x = jnp.array([1.0, -1.0, 0.0, 2.0, -2.0])
        result = _RLion.smooth_sign(x, beta=1.0)

        # tanh should be smooth and bounded between -1 and 1
        self.assertTrue(jnp.all(jnp.abs(result) <= 1.0))
        # Check it's close to sign for large values
        self.assertAlmostEqual(float(result[3]), 1.0, places=1)
        self.assertAlmostEqual(float(result[4]), -1.0, places=1)
        # Check it's 0 at 0
        self.assertAlmostEqual(float(result[2]), 0.0, places=6)

    @chex.all_variants(with_pmap=False)
    def test_rlion_with_weight_decay(self):
        """Test RLion with weight decay."""
        params = self.initial_params
        optimizer = _RLion.rlion(
            learning_rate=0.1,
            b1=0.9,
            b2=0.99,
            weight_decay=0.01,
            use_smooth_sign=True
        )

        final_params, final_state = self.loop(
            optimizer=optimizer,
            num_steps=3,
            params=params
        )

        self.assertEqual(final_state[0].count, 3)
        chex.assert_tree_all_finite((final_params, final_state))

    @chex.all_variants(with_pmap=False)
    def test_rlion_different_beta(self):
        """Test RLion with different smooth_beta values."""
        params = self.initial_params

        # Test with higher beta (sharper transition)
        optimizer = _RLion.rlion(
            learning_rate=0.1,
            b1=0.9,
            b2=0.99,
            use_smooth_sign=True,
            smooth_beta=5.0
        )

        final_params, final_state = self.loop(
            optimizer=optimizer,
            num_steps=3,
            params=params
        )

        self.assertEqual(final_state[0].count, 3)
        chex.assert_tree_all_finite((final_params, final_state))


if __name__ == '__main__':
    absltest.main()
