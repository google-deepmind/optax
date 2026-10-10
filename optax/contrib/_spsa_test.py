# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the SPSA optimizer in `optax.contrib._spsa`."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from optax import apply_updates
from optax.contrib import _spsa


class SPSATest(absltest.TestCase):

  def test_spsa_minimizes_quadratic(self):
    # SPSA only sees objective values, never gradients.
    target = jnp.array([1.0, -2.0, 3.0, 0.5])
    obj_fn = lambda params: 0.5 * jnp.sum((params - target) ** 2)

    params = jnp.zeros_like(target)
    opt = _spsa.spsa(learning_rate=0.2, c=0.1, seed=jax.random.key(0))
    state = opt.init(params)

    @jax.jit
    def step(params, state):
      # `grads` are ignored by SPSA; pass zeros to prove they are unused.
      updates, state = opt.update(
          jnp.zeros_like(params), state, params, obj_fn=obj_fn
      )
      return apply_updates(params, updates), state

    initial_loss = obj_fn(params)
    for _ in range(3000):
      params, state = step(params, state)

    self.assertLess(float(obj_fn(params)), 1e-3)
    self.assertLess(float(obj_fn(params)), float(initial_loss))
    np.testing.assert_allclose(params, target, atol=2e-2)

  def test_spsa_minimizes_quadratic_with_pytree_params(self):
    target = {'w': jnp.array([2.0, -1.0]), 'b': jnp.array(0.5)}
    obj_fn = lambda p: 0.5 * (
        jnp.sum((p['w'] - target['w']) ** 2) + (p['b'] - target['b']) ** 2
    )

    params = {'w': jnp.zeros(2), 'b': jnp.zeros(())}
    opt = _spsa.spsa(learning_rate=0.2, c=0.1, seed=jax.random.key(1))
    state = opt.init(params)

    @jax.jit
    def step(params, state):
      updates, state = opt.update(params, state, params, obj_fn=obj_fn)
      return apply_updates(params, updates), state

    for _ in range(3000):
      params, state = step(params, state)

    self.assertLess(float(obj_fn(params)), 1e-3)

  def test_gradient_estimate_is_unbiased(self):
    # For f(x) = 0.5||x - t||^2 the SPSA estimate is exactly unbiased (no
    # O(c^2) bias term), so averaging many estimates recovers the true
    # gradient `x - t`.
    target = jnp.array([0.7, -1.3, 2.1])
    obj_fn = lambda params: 0.5 * jnp.sum((params - target) ** 2)
    point = jnp.array([1.0, 1.0, 1.0])
    true_grad = point - target

    estimator = _spsa.spsa_gradient(c=0.05, seed=jax.random.key(2))
    state = estimator.init(point)

    @jax.jit
    def one_estimate(state):
      grad, state = estimator.update(point, state, point, obj_fn=obj_fn)
      return grad, state

    acc = jnp.zeros_like(point)
    n = 20000
    for _ in range(n):
      grad, state = one_estimate(state)
      acc = acc + grad
    mean_estimate = acc / n

    np.testing.assert_allclose(mean_estimate, true_grad, atol=5e-2)

  def test_update_requires_obj_fn_and_params(self):
    estimator = _spsa.spsa_gradient()
    params = jnp.ones(3)
    state = estimator.init(params)

    with self.assertRaises(ValueError):
      estimator.update(params, state, params)  # missing obj_fn
    with self.assertRaises(ValueError):
      estimator.update(params, state, obj_fn=jnp.sum)  # no params

  def test_state_count_increments(self):
    estimator = _spsa.spsa_gradient()
    params = jnp.ones(2)
    state = estimator.init(params)
    self.assertEqual(int(state.count), 0)
    _, state = estimator.update(
        params, state, params, obj_fn=jnp.sum
    )
    self.assertEqual(int(state.count), 1)
    self.assertEqual(state.count.dtype, jnp.int32)


if __name__ == '__main__':
  absltest.main()
