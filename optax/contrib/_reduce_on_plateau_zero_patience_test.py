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
"""Zero patience should reduce on bad metrics, never on improvements."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import optax


class ZeroPatienceTest(parameterized.TestCase):

  @parameterized.parameters(
      {'values': [5.0, 4.0, 3.0], 'scales': [1.0, 1.0, 1.0]},
      {'values': [3.0, 3.0, 3.0], 'scales': [1.0, 0.5, 0.25]},
      {'values': [3.0, 4.0, 5.0], 'scales': [1.0, 0.5, 0.25]},
      {
          'values': [5.0, 6.0, 4.0, 5.0, 3.0],
          'scales': [1.0, 0.5, 0.5, 0.25, 0.25],
      },
  )
  def test_zero_patience_reduces_only_after_non_improvement(
      self, values, scales
  ):
    for compiled in [False, True]:
      tx = optax.contrib.reduce_on_plateau(patience=0, factor=0.5)
      params = jnp.ones(2)
      state = tx.init(params)
      update = jax.jit(tx.update) if compiled else tx.update
      for value, scale in zip(values, scales):
        updates, state = update(
            jnp.ones_like(params), state, value=jnp.array(value)
        )
        np.testing.assert_allclose(updates, scale, rtol=0, atol=0)
        self.assertEqual(float(state.scale), scale)
        self.assertEqual(int(state.plateau_count), 0)

  def test_cooldown_starts_on_first_bad_metric(self):
    tx = optax.contrib.reduce_on_plateau(patience=0, factor=0.5, cooldown=2)
    state = tx.init(jnp.ones(1))
    for scale, cooldown in [(1.0, 0), (0.5, 2), (0.5, 1), (0.5, 0), (0.25, 2)]:
      _, state = jax.jit(tx.update)(jnp.ones(1), state, value=jnp.array(3.0))
      self.assertEqual(float(state.scale), scale)
      self.assertEqual(int(state.cooldown_count), cooldown)

  def test_accumulation_defers_decisions_until_metric_is_complete(self):
    tx = optax.contrib.reduce_on_plateau(
        patience=jnp.array(0), factor=0.5, accumulation_size=2
    )
    state = tx.init({'w': jnp.ones(2)})
    for value, scale in zip(
        [3.0, 3.0, 3.0, 3.0, 2.0, 2.0], [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    ):
      updates, state = jax.jit(tx.update)(
          {'w': jnp.ones(2)}, state, value=jnp.array(value)
      )
      np.testing.assert_allclose(updates['w'], scale)

  def test_repeated_plateaus_stop_at_minimum_scale(self):
    tx = optax.contrib.reduce_on_plateau(patience=0, factor=0.5, min_scale=0.2)
    state = tx.init(jnp.ones(1))
    for scale in [1.0, 0.5, 0.25, 0.2, 0.2]:
      _, state = tx.update(jnp.ones(1), state, value=jnp.array(3.0))
      np.testing.assert_allclose(state.scale, scale)

  @parameterized.parameters(1, 2, 5)
  def test_positive_patience_retains_existing_wait_count(self, patience):
    tx = optax.contrib.reduce_on_plateau(patience=patience, factor=0.5)
    state = tx.init(jnp.ones(1))
    _, state = tx.update(jnp.ones(1), state, value=jnp.array(3.0))
    for step in range(1, patience + 1):
      _, state = tx.update(jnp.ones(1), state, value=jnp.array(3.0))
      self.assertEqual(float(state.scale), 0.5 if step == patience else 1.0)

  def test_chained_sgd_keeps_full_step_during_improvement(self):
    tx = optax.chain(
        optax.sgd(0.1), optax.contrib.reduce_on_plateau(patience=0, factor=0.5)
    )
    params = jnp.array([1.0])
    state = tx.init(params)
    for value, update in [(5.0, -0.1), (4.0, -0.1), (4.0, -0.05)]:
      updates, state = jax.jit(tx.update)(
          jnp.ones(1), state, params, value=jnp.array(value)
      )
      np.testing.assert_allclose(updates, update, rtol=2e-6)
      params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(params, [0.75], rtol=2e-6)


if __name__ == '__main__':
  absltest.main()
