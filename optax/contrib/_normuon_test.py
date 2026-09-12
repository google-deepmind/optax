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
"""Tests for the NorMuon optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _normuon


class NorMuonTest(parameterized.TestCase):

  def test_basic_normuon(self):
    """Test that normuon() runs and produces finite outputs."""
    key = jax.random.key(0)
    params = {'w': jax.random.normal(key, (8, 6))}
    opt = _normuon.normuon(learning_rate=1e-3)
    state = opt.init(params)
    grad = params
    updates, new_state = opt.update(grad, state, params=params)
    self.assertEqual(updates['w'].shape, (8, 6))
    self.assertTrue(jnp.all(jnp.isfinite(updates['w'])))
    del new_state

  def test_mixed_params(self):
    """Test that 2D params go through NorMuon and 1D through Adam."""
    key = jax.random.key(1)
    params = {
        'w': jax.random.normal(key, (10, 5)),
        'b': jax.random.normal(key, (5,)),
    }
    opt = _normuon.normuon(learning_rate=1e-3)
    state = opt.init(params)
    grad = params
    updates, _ = opt.update(grad, state, params=params)
    self.assertEqual(updates['w'].shape, (10, 5))
    self.assertEqual(updates['b'].shape, (5,))
    self.assertTrue(jnp.all(jnp.isfinite(updates['w'])))
    self.assertTrue(jnp.all(jnp.isfinite(updates['b'])))

  def test_convergence(self):
    """Test that NorMuon can optimize a simple quadratic."""
    key = jax.random.key(2)
    target = jax.random.normal(key, (4, 4))

    def loss_fn(params):
      return jnp.sum((params['w'] - target) ** 2)

    opt = _normuon.normuon(learning_rate=1e-2)
    params = {'w': jnp.zeros((4, 4))}
    state = opt.init(params)

    initial_loss = loss_fn(params)
    for _ in range(200):
      grad = jax.grad(loss_fn)(params)
      updates, state = opt.update(grad, state, params=params)
      params = update.apply_updates(params, updates)

    final_loss = loss_fn(params)
    self.assertLess(final_loss, initial_loss * 0.5)

  @parameterized.product(
      shape=[(6, 4), (8, 8), (3, 10)],
  )
  def test_scale_by_normuon_direct(self, shape):
    """Test scale_by_normuon directly on a 2D input."""
    key = jax.random.key(3)
    params = jax.random.normal(key, shape)
    opt = _normuon.scale_by_normuon()
    state = opt.init(params)
    grad = jax.random.normal(key, shape)
    updates, new_state = opt.update(grad, state)
    self.assertEqual(updates.shape, shape)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))
    # Check that nu state has the right shape (rows only).
    self.assertEqual(new_state.nu.shape, (shape[0],))

  @parameterized.named_parameters(
      ('small', 1e-7),
      ('large', 1e7),
  )
  def test_numerical_stability(self, scale):
    """Test NorMuon with very small and very large inputs."""
    key = jax.random.key(4)
    params = jax.random.normal(key, (8, 4)) * scale
    opt = _normuon.normuon(learning_rate=1e-3)
    state = opt.init(params)
    grad = params
    updates, _ = opt.update(grad, state, params=params)
    self.assertTrue(
        jnp.all(jnp.isfinite(updates['w']))
        if isinstance(updates, dict)
        else jnp.all(jnp.isfinite(updates))
    )

  def test_normuon_state_structure(self):
    """Test that NorMuonState has the expected fields."""
    params = jnp.ones((4, 3))
    opt = _normuon.scale_by_normuon()
    state = opt.init(params)
    self.assertIsInstance(state, _normuon.NorMuonState)
    self.assertEqual(state.count, 0)
    self.assertEqual(state.mu.shape, (4, 3))
    self.assertEqual(state.nu.shape, (4,))

  def test_nesterov_flag(self):
    """Test that nesterov=True and False produce different momentum states."""
    key = jax.random.key(5)
    params = jax.random.normal(key, (6, 4))

    opt_nest = _normuon.scale_by_normuon(nesterov=True)
    opt_no_nest = _normuon.scale_by_normuon(nesterov=False)

    state_nest = opt_nest.init(params)
    state_no_nest = opt_no_nest.init(params)

    grad = jax.random.normal(jax.random.key(6), (6, 4))

    # Run two steps so the momentum accumulates differently.
    _, state_nest = opt_nest.update(grad, state_nest)
    _, state_no_nest = opt_no_nest.update(grad, state_no_nest)

    grad2 = jax.random.normal(jax.random.key(7), (6, 4))
    updates_nest, _ = opt_nest.update(grad2, state_nest)
    updates_no_nest, _ = opt_no_nest.update(grad2, state_no_nest)

    # The nu states should be identical (same ortho output), but the
    # momentum paths differ, leading to different ortho inputs and thus
    # different final updates.
    self.assertFalse(jnp.allclose(updates_nest, updates_no_nest, atol=1e-6))


if __name__ == '__main__':
  absltest.main()
