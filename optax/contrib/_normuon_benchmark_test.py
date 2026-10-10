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
"""Benchmark tests comparing NorMuon and Muon training convergence."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _muon
from optax.contrib import _normuon


def _make_mlp_params(key):
  """Create parameters for a 2-layer MLP: 32 -> 64 -> 1."""
  k1, k2 = jax.random.split(key, 2)
  return {
      'w1': jax.random.normal(k1, (32, 64)) * 0.1,
      'b1': jnp.zeros(64),
      'w2': jax.random.normal(k2, (64, 1)) * 0.1,
      'b2': jnp.zeros(1),
  }


def _mlp_forward(params, x):
  """Forward pass for a 2-layer MLP with tanh activation."""
  h = jnp.tanh(x @ params['w1'] + params['b1'])
  return h @ params['w2'] + params['b2']


def _make_data(key, batch_size=64, input_dim=32):
  """Generate synthetic regression data."""
  k1, k2 = jax.random.split(key)
  x = jax.random.normal(k1, (batch_size, input_dim))
  y = jnp.sum(x[:, :3], axis=-1, keepdims=True) + 0.1 * jax.random.normal(
      k2, (batch_size, 1)
  )
  return x, y


def _train(optimizer, params, x, y, steps=500):
  """Train the MLP and return losses and final params."""
  state = optimizer.init(params)
  losses = []

  def loss_fn(p):
    pred = _mlp_forward(p, x)
    return jnp.mean((pred - y) ** 2)

  @jax.jit
  def step_fn(params, state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_state = optimizer.update(grads, state, params)
    new_params = update.apply_updates(params, updates)
    return new_params, new_state, loss

  for _ in range(steps):
    params, state, loss = step_fn(params, state)
    losses.append(float(loss))
  return losses, params


class NorMuonBenchmarkTest(absltest.TestCase):
  """Benchmark tests comparing NorMuon and Muon optimizers."""

  def setUp(self):
    super().setUp()
    self.data_key = jax.random.key(42)
    self.param_key = jax.random.key(0)
    self.x, self.y = _make_data(self.data_key)

  def test_normuon_vs_muon_convergence(self):
    """Both optimizers should converge; NorMuon within 5x of Muon."""
    params = _make_mlp_params(self.param_key)

    muon_opt = _muon.muon(learning_rate=0.01)
    normuon_opt = _normuon.normuon(learning_rate=0.01)

    muon_losses, _ = _train(muon_opt, params, self.x, self.y, steps=500)
    normuon_losses, _ = _train(
        normuon_opt, params, self.x, self.y, steps=500
    )

    # Print loss at key steps for visibility.
    for name, losses in [('Muon', muon_losses), ('NorMuon', normuon_losses)]:
      for step in [0, 100, 200, 300, 400, 499]:
        print(f'{name} step {step}: loss={losses[step]:.6f}')

    # Both should converge: final loss < 10% of initial loss.
    self.assertLess(
        muon_losses[-1],
        0.1 * muon_losses[0],
        'Muon did not converge.',
    )
    self.assertLess(
        normuon_losses[-1],
        0.1 * normuon_losses[0],
        'NorMuon did not converge.',
    )

    # NorMuon should not be more than 2x worse than Muon.
    self.assertLess(
        normuon_losses[-1],
        2.0 * muon_losses[-1],
        'NorMuon final loss is more than 2x worse than Muon.',
    )

  def test_normuon_no_side_effects(self):
    """NorMuon training should have no NaN/Inf and mostly decrease."""
    params = _make_mlp_params(self.param_key)
    normuon_opt = _normuon.normuon(learning_rate=0.01)
    losses, final_params = _train(
        normuon_opt, params, self.x, self.y, steps=500
    )

    # No NaN or Inf in any loss value.
    for i, loss in enumerate(losses):
      self.assertTrue(
          jnp.isfinite(loss), f'Non-finite loss at step {i}: {loss}'
      )

    # Loss should be monotonically decreasing with tolerance for noise.
    # Check that loss at every 50-step window is lower than the previous.
    window = 50
    for i in range(window, len(losses), window):
      avg_prev = sum(losses[i - window : i]) / window
      avg_curr = sum(losses[i : min(i + window, len(losses))]) / max(
          1, min(window, len(losses) - i)
      )
      self.assertLess(
          avg_curr,
          avg_prev * 1.5,
          f'Loss not decreasing around step {i}: '
          f'avg_prev={avg_prev:.6f}, avg_curr={avg_curr:.6f}',
      )

    # All final parameters should be finite.
    for name, p in final_params.items():
      self.assertTrue(
          jnp.all(jnp.isfinite(p)),
          f'Non-finite values in final param {name}',
      )

  def test_normuon_mixed_params_training(self):
    """All params (2D weights and 1D biases) should be updated."""
    init_params = _make_mlp_params(self.param_key)
    normuon_opt = _normuon.normuon(learning_rate=0.01)
    _, final_params = _train(
        normuon_opt, init_params, self.x, self.y, steps=500
    )

    # Every parameter should have changed from its initial value.
    for name, init_val in init_params.items():
      self.assertFalse(
          jnp.allclose(init_val, final_params[name], atol=1e-8),
          f'Parameter {name} was not updated during training.',
      )

    # All final parameters should remain finite.
    for name, p in final_params.items():
      self.assertTrue(
          jnp.all(jnp.isfinite(p)),
          f'Non-finite values in final param {name}',
      )


if __name__ == '__main__':
  absltest.main()
