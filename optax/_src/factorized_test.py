# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `factorized.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

import optax
from optax._src import factorized


def replicate(tree):
  """Replicates arrays to multiple devices."""
  return jax.device_put_replicated(tree, jax.local_devices())


class FactorizedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants
  def test_scale_by_factored_rms(self):
    params = self.init_params

    scaler = factorized.scale_by_factored_rms()
    init_fn = self.variant(scaler.init)
    transform_fn = self.variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    updates, state = transform_fn(self.per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))
    chex.assert_trees_all_equal_shapes(params, updates)

  def test_scale_by_factored_rms_with_injected_hyperparams(self):
    params = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))
    tx = optax.inject_hyperparams(optax.adafactor)(learning_rate=0.0)
    opt_state = tx.init(params)
    replicated_params = replicate(params)
    replicated_opt_state = replicate(opt_state)

    def _update(params, opt_state):
      updates = jax.random.uniform(
          jax.random.PRNGKey(jax.lax.axis_index('device')), (10, 10))
      updates, opt_state = tx.update(updates, opt_state, params)
      params = optax.apply_updates(params, updates)
      params = jax.lax.pmean(params, axis_name='device')
      return params, opt_state

    jax.pmap(_update, 'device')(replicated_params, replicated_opt_state)


if __name__ == '__main__':
  absltest.main()
