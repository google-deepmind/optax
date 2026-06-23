# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for optax.contrib.spsa."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import optax


class SPSATest(chex.TestCase):
    @chex.all_variants
    def test_spsa_gradient_estimator(self):
        def loss_fn(params):
            return jnp.sum(params["w"] ** 2)

        estimator = optax.contrib.spsa_estimator(loss_fn)
        params = {"w": jnp.array([1.0, -2.0, 3.0])}

        # We use a fixed key to ensure deterministic test.
        key = jax.random.PRNGKey(42)

        @self.variant
        def get_grad(p, k, c):
            return estimator(p, c, k)

        # Average over 10000 samples to test unbiasedness
        keys = jax.random.split(key, 10000)
        get_grad_vmap = jax.vmap(get_grad, in_axes=(None, 0, None))
        grad_estimates = get_grad_vmap(params, keys, 1.0)

        mean_grad = jax.tree.map(lambda x: jnp.mean(x, axis=0), grad_estimates)
        expected_grad = {"w": jnp.array([2.0, -4.0, 6.0])}

        # Check if the mean of the estimates is close to the true gradient
        chex.assert_trees_all_close(mean_grad, expected_grad, atol=0.2)

    @chex.all_variants
    def test_spsa_optimizer_integration(self):
        # Test using SPSA estimator with standard optax optimizer (e.g. SGD)
        def loss_fn(params):
            return jnp.sum((params["w"] - 2.0) ** 2)

        estimator = optax.contrib.spsa_estimator(loss_fn)
        optimizer = optax.sgd(learning_rate=0.1)

        params = {"w": jnp.array([-5.0, 5.0])}
        opt_state = optimizer.init(params)
        key = jax.random.PRNGKey(0)

        @self.variant
        def step(p, state, k):
            k1, k2 = jax.random.split(k)
            grad = estimator(p, 0.1, k1)
            updates, new_state = optimizer.update(grad, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, new_state, k2

        for _ in range(50):
            params, opt_state, key = step(params, opt_state, key)

        # After 50 steps, parameters should be close to 2.0
        chex.assert_trees_all_close(
            params["w"], jnp.array([2.0, 2.0]), atol=1e-2
        )

    def test_spsa_standard_schedule(self):
        schedule = optax.contrib.spsa_standard_schedule(
            init_value=1.0, decay_rate=0.5, offset=10.0
        )

        val_0 = schedule(0)
        val_10 = schedule(10)

        self.assertAlmostEqual(val_0, 1.0 / (10.0**0.5))
        self.assertAlmostEqual(val_10, 1.0 / (20.0**0.5))


if __name__ == "__main__":
    absltest.main()
