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
"""Tests for the Prodigy optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _prodigy


class ProdigyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _prodigy.prodigy()
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_stores_params0(self):
    """Prodigy stores initial params to compute distance to solution."""
    opt = _prodigy.prodigy()
    state = opt.init(self.params)
    jnp.testing.assert_array_equal(state.params0, self.params)

  def test_state_init_shapes(self):
    opt = _prodigy.prodigy()
    state = opt.init(self.params)
    self.assertEqual(state.exp_avg.shape, self.params.shape)
    self.assertEqual(state.exp_avg_sq.shape, self.params.shape)
    self.assertEqual(state.grad_sum.shape, self.params.shape)
    self.assertEqual(state.estim_lr.shape, ())
    self.assertEqual(state.count.shape, ())

  def test_single_step_finite(self):
    opt = _prodigy.prodigy()
    state = opt.init(self.params)
    updates, new_state = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_requires_params(self):
    opt = _prodigy.prodigy()
    state = opt.init(self.params)
    with self.assertRaises(ValueError):
      opt.update(self.grads, state, params=None)

  def test_estim_lr_non_decreasing(self):
    """The estimated learning rate should be non-decreasing."""
    opt = _prodigy.prodigy(learning_rate=1.0)
    state = opt.init(self.params)
    prev_estim_lr = state.estim_lr
    for _ in range(10):
      _, state = opt.update(self.grads, state, self.params)
      self.assertGreaterEqual(float(state.estim_lr), float(prev_estim_lr))
      prev_estim_lr = state.estim_lr

  @parameterized.product(
      weight_decay=(0.0, 0.01),
      safeguard_warmup=(True, False),
  )
  def test_options(self, weight_decay, safeguard_warmup):
    opt = _prodigy.prodigy(
        weight_decay=weight_decay, safeguard_warmup=safeguard_warmup
    )
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
