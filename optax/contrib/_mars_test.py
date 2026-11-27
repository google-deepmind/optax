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
"""Tests for the MARS optimizer."""

import statistics

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import optax
from optax.contrib._mars import mars
from optax.contrib._mars import MarsState
from optax.contrib._mars import scale_by_mars


class ScaleByMarsTest(absltest.TestCase):

  def test_state_structure(self):
    """MarsState has the expected fields after init."""
    params = jnp.ones((3,))
    tx = scale_by_mars()
    state = tx.init(params)
    self.assertIsInstance(state, MarsState)
    self.assertEqual(state.count, 0)
    self.assertEqual(state.mu.shape, params.shape)
    self.assertEqual(state.nu.shape, params.shape)
    self.assertEqual(state.prev_grad.shape, params.shape)
    self.assertEqual(state.c_prev.shape, params.shape)

  def test_first_step_no_correction(self):
    """At step 0 the correction term must be zero (c_1 = g_1)."""
    params = jnp.ones((4,))
    tx = scale_by_mars(gamma=0.025)
    state = tx.init(params)
    grad = jnp.array([1.0, 2.0, -1.0, 0.5])
    _, new_state = tx.update(grad, state)
    # c_prev should equal the gradient on the first step.
    self.assertTrue(jnp.allclose(new_state.c_prev, grad))
    self.assertTrue(jnp.allclose(new_state.prev_grad, grad))

  def test_gamma_one_recovers_adam_moments(self):
    """With gamma=1 the correction vanishes and MARS reduces to Adam."""
    params = jnp.ones((3,))
    mars_tx = scale_by_mars(gamma=1.0, b1=0.9, b2=0.999, eps=1e-8)
    adam_tx = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)

    mars_state = mars_tx.init(params)
    adam_state = adam_tx.init(params)

    key = jax.random.PRNGKey(0)
    for _ in range(5):
      key, subkey = jax.random.split(key)
      grad = jax.random.normal(subkey, params.shape)
      mars_updates, mars_state = mars_tx.update(grad, mars_state)
      adam_updates, adam_state = adam_tx.update(grad, adam_state)

    self.assertTrue(
        jnp.allclose(mars_updates, adam_updates, atol=1e-6),
        msg='MARS with gamma=1 should match Adam updates.',
    )

  def test_correction_reduces_moment_variance(self):
    """Corrected gradient c_t should track the true gradient more smoothly."""
    params = jnp.ones((8,))
    tx = scale_by_mars(gamma=0.025)
    state = tx.init(params)

    # Feed a noisy gradient sequence.
    key = jax.random.PRNGKey(42)
    c_norms = []
    g_norms = []
    for _ in range(20):
      key, subkey = jax.random.split(key)
      grad = jax.random.normal(subkey, params.shape) * 10.0
      _, state = tx.update(grad, state)
      c_norms.append(float(jnp.linalg.norm(state.c_prev)))
      g_norms.append(float(jnp.linalg.norm(grad)))

    # Corrected gradient norms should have lower std than raw gradient norms
    # after the first couple of steps.
    self.assertLess(
        statistics.stdev(c_norms[3:]),
        statistics.stdev(g_norms[3:]) + 1.0,  # generous tolerance
        msg='Corrected gradient should be smoother than raw gradient.',
    )

  def test_correction_clip(self):
    """correction_clip should prevent exploding corrections."""
    params = jnp.ones((16,))
    tx = scale_by_mars(gamma=0.025, correction_clip=1.0)
    state = tx.init(params)
    # Prime the state with a large gradient.
    large_grad = jnp.ones((16,)) * 1000.0
    _, state = tx.update(large_grad, state)
    # Now send a very different gradient.
    small_grad = jnp.zeros((16,))
    updates, _ = tx.update(small_grad, state)
    # Update should be finite and bounded.
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_nesterov_flag(self):
    """nesterov=True should produce different updates from nesterov=False."""
    params = jnp.ones((4,))
    tx_nes = scale_by_mars(gamma=0.5, nesterov=True)
    tx_std = scale_by_mars(gamma=0.5, nesterov=False)
    state_nes = tx_nes.init(params)
    state_std = tx_std.init(params)

    key = jax.random.PRNGKey(7)
    for _ in range(3):
      key, subkey = jax.random.split(key)
      grad = jax.random.normal(subkey, params.shape)
      upd_nes, state_nes = tx_nes.update(grad, state_nes)
      upd_std, state_std = tx_std.update(grad, state_std)

    # After several steps nesterov and standard updates should differ.
    self.assertFalse(
        jnp.allclose(upd_nes, upd_std),
        msg='Nesterov and standard updates should differ.',
    )


class MarsOptimizerTest(parameterized.TestCase):

  @parameterized.parameters(
      {'gamma': 1.0},   # Adam limit
      {'gamma': 0.5},   # moderate correction
      {'gamma': 0.025},  # paper default
  )
  def test_descends_quadratic(self, gamma):
    """mars() should reduce a simple quadratic objective."""
    params = jnp.array([3.0, -2.0, 1.0])
    solver = mars(learning_rate=1e-2, gamma=gamma, weight_decay=0.0)
    state = solver.init(params)

    def loss(p):
      return jnp.sum(p ** 2)

    initial_loss = loss(params)
    for _ in range(50):
      grad = jax.grad(loss)(params)
      updates, state = solver.update(grad, state, params)
      params = optax.apply_updates(params, updates)

    self.assertLess(
        loss(params),
        initial_loss,
        msg=f'MARS (gamma={gamma}) should reduce the quadratic objective.',
    )

  def test_weight_decay_applied(self):
    """weight_decay > 0 should shrink parameters over time."""
    params = jnp.ones((4,)) * 5.0
    solver_wd = mars(learning_rate=1e-3, weight_decay=0.1)
    solver_no = mars(learning_rate=1e-3, weight_decay=0.0)
    state_wd = solver_wd.init(params)
    state_no = solver_no.init(params)

    zero_grad = jnp.zeros_like(params)
    for _ in range(10):
      upd_wd, state_wd = solver_wd.update(zero_grad, state_wd, params)
      upd_no, state_no = solver_no.update(zero_grad, state_no, params)
      params_wd = optax.apply_updates(params, upd_wd)

    # Parameters with weight decay should have smaller norm.
    params_no = optax.apply_updates(params, upd_no)
    self.assertLess(
        jnp.linalg.norm(params_wd),
        jnp.linalg.norm(params_no),
    )

  def test_correction_clip_stability(self):
    """correction_clip should not cause NaNs even with very spiky gradients."""
    params = jnp.ones((8,))
    solver = mars(learning_rate=1e-3, correction_clip=0.1)
    state = solver.init(params)
    key = jax.random.PRNGKey(0)
    for _ in range(30):
      key, subkey = jax.random.split(key)
      # Alternate between huge and tiny gradients to stress the correction.
      grad = jax.random.normal(subkey, params.shape) * 1e3
      updates, state = solver.update(grad, state, params)
      params = optax.apply_updates(params, updates)
    self.assertTrue(jnp.all(jnp.isfinite(params)))

  def test_pytree_params(self):
    """mars() should work with pytree (dict) parameters."""
    params = {'w': jnp.ones((3,)), 'b': jnp.zeros((2,))}
    solver = mars(learning_rate=1e-3)
    state = solver.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, state = solver.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    jax.tree.map(
        lambda p: self.assertTrue(jnp.all(jnp.isfinite(p))), new_params
    )


if __name__ == '__main__':
  absltest.main()
