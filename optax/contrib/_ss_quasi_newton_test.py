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
"""Tests for self-scaled quasi-Newton optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.contrib import _ss_quasi_newton as ssqn


class SSQuasiNewtonTest(parameterized.TestCase):

  def test_quadratic_convergence(self):
    a = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, 2.0])

    def loss_fn(params):
      x = params["w"]
      return 0.5 * jnp.dot(x, jnp.matmul(a, x)) - jnp.dot(b, x)

    # Optimal solution x* = A^{-1} b
    optimal_x = jnp.linalg.solve(a, b)

    opt = ssqn.ssbfgs(learning_rate=1.0)
    params = {"w": jnp.array([10.0, -5.0])}
    state = opt.init(params)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    for _ in range(15):
      val, grads = value_and_grad_fn(params)
      updates, state = opt.update(
          grads,
          state,
          params,
          value=val,
          grad=grads,
          value_fn=loss_fn,
      )
      params = optax.apply_updates(params, updates)

    np.testing.assert_allclose(params["w"], optimal_x, atol=1e-4)

  def test_rosenbrock_convergence(self):
    def rosenbrock(params):
      x, y = params["x"], params["y"]
      return jnp.sum(100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2)

    opt = ssqn.ssbroyden(phi=0.5, learning_rate=1.0)
    params = {"x": jnp.array(-1.2), "y": jnp.array(1.0)}
    state = opt.init(params)

    value_and_grad_fn = jax.value_and_grad(rosenbrock)

    for _ in range(60):
      val, grads = value_and_grad_fn(params)
      updates, state = opt.update(
          grads,
          state,
          params,
          value=val,
          grad=grads,
          value_fn=rosenbrock,
      )
      params = optax.apply_updates(params, updates)

    np.testing.assert_allclose(params["x"], 1.0, atol=1e-2)
    np.testing.assert_allclose(params["y"], 1.0, atol=1e-2)

  def test_pytree_support(self):
    params = {
        "linear": {"w": jnp.ones((2, 2)), "b": jnp.zeros((2,))},
        "scalar": jnp.array(1.0),
    }
    opt = ssqn.scale_by_ss_quasi_newton()
    state = opt.init(params)

    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, next_state = opt.update(grads, state, params=params)

    # Shapes must match original PyTree leaves
    chex.assert_trees_all_equal_shapes(params, updates)
    self.assertEqual(next_state.count, 1)


if __name__ == "__main__":
  absltest.main()
