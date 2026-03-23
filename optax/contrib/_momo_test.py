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
"""Tests for the MoMo and MoMo-Adam optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _momo


class MomoTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.obj_fn = lambda p: jnp.sum(p**2)
    self.value, self.grads = jax.value_and_grad(self.obj_fn)(self.params)

  def test_state_init(self):
    opt = _momo.momo()
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_single_step_finite(self):
    opt = _momo.momo()
    state = opt.init(self.params)
    updates, new_state = opt.update(
        self.grads, state, self.params, value=self.value
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_requires_params(self):
    opt = _momo.momo()
    state = opt.init(self.params)
    with self.assertRaises(ValueError):
      opt.update(self.grads, state, params=None, value=self.value)

  def test_requires_value(self):
    opt = _momo.momo()
    state = opt.init(self.params)
    with self.assertRaises((ValueError, TypeError)):
      opt.update(self.grads, state, self.params)

  @parameterized.product(adapt_lower_bound=(True, False))
  def test_adapt_lower_bound(self, adapt_lower_bound):
    opt = _momo.momo(adapt_lower_bound=adapt_lower_bound)
    state = opt.init(self.params)
    updates, _ = opt.update(
        self.grads, state, self.params, value=self.value
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_zero_gradients_zero_loss(self):
    opt = _momo.momo()
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(
        zero_grads, state, self.params, value=jnp.array(0.0)
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


class MomoAdamTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.obj_fn = lambda p: jnp.sum(p**2)
    self.value, self.grads = jax.value_and_grad(self.obj_fn)(self.params)

  def test_state_init(self):
    opt = _momo.momo_adam()
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_has_exp_avg_sq(self):
    """MoMo-Adam should have second moment estimate unlike plain MoMo."""
    opt = _momo.momo_adam()
    state = opt.init(self.params)
    self.assertEqual(state.exp_avg_sq.shape, self.params.shape)

  def test_single_step_finite(self):
    opt = _momo.momo_adam()
    state = opt.init(self.params)
    updates, _ = opt.update(
        self.grads, state, self.params, value=self.value
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_requires_value(self):
    opt = _momo.momo_adam()
    state = opt.init(self.params)
    with self.assertRaises((ValueError, TypeError)):
      opt.update(self.grads, state, self.params)

  @parameterized.product(weight_decay=(0.0, 0.01))
  def test_weight_decay(self, weight_decay):
    opt = _momo.momo_adam(weight_decay=weight_decay)
    state = opt.init(self.params)
    updates, _ = opt.update(
        self.grads, state, self.params, value=self.value
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
