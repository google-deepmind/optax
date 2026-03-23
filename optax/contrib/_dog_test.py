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
"""Tests for the DoG (Distance over Gradients) optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _dog


class DogTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jnp.array([0.1, 0.2, 0.3])

  def test_state_init(self):
    opt = _dog.dog()
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_state_stores_init_params(self):
    """DoG stores initial params to compute distance."""
    opt = _dog.scale_by_dog(init_step=('heuristic', 1e-6))
    state = opt.init(self.params)
    jnp.testing.assert_array_equal(state.init_params, self.params)

  def test_single_step_finite(self):
    opt = _dog.dog()
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  @parameterized.parameters(
      ('heuristic', 1e-6),
      ('distance', 1e-4),
      ('learning_rate', 1e-3),
  )
  def test_init_step_types(self, init_type, init_value):
    opt = _dog.dog(init_step=(init_type, init_value))
    state = opt.init(self.params)
    updates, _ = opt.update(self.grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_max_dist_grows(self):
    """max_dist should grow as params move away from init."""
    opt = _dog.scale_by_dog(init_step=('distance', 0.0))
    state = opt.init(self.params)
    params = self.params
    for _ in range(5):
      updates, state = opt.update(self.grads, state, params)
      params = update.apply_updates(params, updates)
    self.assertGreater(float(state.max_dist), 0.0)

  def test_zero_gradients(self):
    opt = _dog.dog()
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(zero_grads, state, self.params)
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
