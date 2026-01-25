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
"""Tests for MADGRAD optimizer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax
from optax.contrib import _madgrad


class MadgradTest(absltest.TestCase):

  def test_quadratic_convergence(self):
    # A simple quadratic function: f(x) = sum(x^2)
    # Minimum is at x = [0, 0, 0]
    params = jnp.array([1.0, 2.0, 3.0])

    # MADGRAD typically needs a slightly higher LR than Adam
    optimizer = _madgrad.madgrad(learning_rate=0.1, momentum=0.9)
    opt_state = optimizer.init(params)

    # Training loop
    for _ in range(500):
      grads = 2 * params  # Gradient of x^2 is 2x
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)

    # Check if we converged close to zero
    self.assertTrue(jnp.allclose(params, jnp.zeros_like(params), atol=1e-3))


if __name__ == '__main__':
  absltest.main()
