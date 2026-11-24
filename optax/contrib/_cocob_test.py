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
"""Numerical regression tests for the COCOB optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax import contrib
from optax._src import test_utils
from optax._src import update
from optax._src import utils
from optax.contrib import _cocob


def _reference_step(params, grads, state):
  """Evaluates the original dimensional COCOB formula in NumPy float64."""
  p0, cumulative, scale, subgradients, reward = state
  scale = np.maximum(scale, np.abs(grads))
  subgradients = subgradients + np.abs(grads)
  reward = np.maximum(reward - grads * (params - p0), 0)
  cumulative = cumulative - grads
  params = p0 + cumulative / (
      scale * np.maximum(subgradients + scale, 100 * scale)
  ) * (scale + reward)
  return params, _cocob.COCOBState(p0, cumulative, scale, subgradients, reward)


class CocobTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float32, jnp.float64),
      power=(-100, 0, 64, 100, 120),
      compiled=(False, True),
  )
  def test_scaled_quadratic(self, dtype, power, compiled):
    with utils.x64_precision(dtype == jnp.float64):
      opt = contrib.cocob()
      params = jnp.zeros(3, dtype)
      state = opt.init(params)
      loss_scale = jnp.asarray(2.0**power, dtype)

      def step(params, state, target):
        loss, grads = jax.value_and_grad(
            lambda x: loss_scale * jnp.sum((x - target) ** 2)
        )(params)
        updates, state = opt.update(grads, state, params)
        return update.apply_updates(params, updates), state, loss, grads

      if compiled:
        step = jax.jit(step)
      expected_params = np.zeros(3, np.float64)
      expected_state = _cocob.COCOBState(
          expected_params.copy(),
          np.zeros(3, np.float64),
          np.full(3, float(np.asarray(1e-8, dtype))),
          np.zeros(3, np.float64),
          np.zeros(3, np.float64),
      )
      tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
      tiny = np.finfo(np.dtype(dtype)).tiny
      for target in ((0, 0, 0), (1, -2, 0), (-1, 2, 0), (1, -2, 0)):
        params, state, loss, grads = step(
            params, state, jnp.asarray(target, dtype)
        )
        expected_params, expected_state = _reference_step(
            expected_params, np.asarray(grads, np.float64), expected_state
        )
        self.assertEqual(params.dtype, dtype)
        for leaf in jax.tree.leaves((params, state, loss, grads)):
          self.assertTrue(np.all(np.isfinite(leaf)))
        np.testing.assert_allclose(
            params, expected_params, rtol=tolerance, atol=tiny
        )
        test_utils.assert_trees_all_close(
            state[-1], expected_state, rtol=tolerance, atol=tiny
        )


if __name__ == '__main__':
  absltest.main()
