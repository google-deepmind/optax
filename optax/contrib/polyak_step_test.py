# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `polyak_step.py`."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import optax


class PolyakStepTest(chex.TestCase):

  # @chex.all_variants
  def test_binary_logistic(self):

    data = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    targets = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (10,))

    def loss(params):
      prediction = data @ params
      return optax.losses.sigmoid_binary_cross_entropy(
          prediction, targets
      ).mean()

    params = jnp.zeros(data.shape[1])

    # make sure initial loss is not too low so we can measure progress
    self.assertGreater(loss(params), 0.6)

    optimizer = optax.contrib.polyak_step_sgd(max_stepsize=100.)
    opt_state = optimizer.init(params)

    assert opt_state is not None

    for _ in range(50):
      value, grad = jax.value_and_grad(loss)(params)
      updates, opt_state = optimizer.update(grad, opt_state, value=value)
      params = optax.apply_updates(params, updates)
      print(value)

    self.assertLess(loss(params), 0.1)

    # Use the chex variant to check various function versions (jit, pmap, etc).


if __name__ == '__main__':
  absltest.main()
