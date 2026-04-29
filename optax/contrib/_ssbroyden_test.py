# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the SSBroyden/SSBFGS optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import base
from optax._src import update
import optax.tree


def _run_opt(opt, fun, init_params, maxiter=500, tol=1e-3):
    """Run a line-search optimizer until convergence."""
    value_and_grad_fun = jax.value_and_grad(fun)

    def stopping_criterion(carry):
        _, _, count, grad = carry
        return (optax.tree.norm(grad) >= tol) & (count < maxiter)

    def step(carry):
        params, state, count, _ = carry
        value, grad = value_and_grad_fun(params)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = update.apply_updates(params, updates)
        return params, state, count + 1, grad

    init_state = opt.init(init_params)
    init_grad = jax.grad(fun)(init_params)
    final_params, final_state, *_ = jax.lax.while_loop(
        stopping_criterion, step, (init_params, init_state, 0, init_grad)
    )
    return final_params, final_state


class SSBroydenTest(parameterized.TestCase):

    @parameterized.parameters("ssbfgs", "ssbroyden")
    def test_quadratic(self, method):
        """Test convergence on a simple quadratic f(x) = sum(x^2)."""
        fun = lambda x: jnp.sum(x**2)
        init_params = jnp.array([1.0, 2.0, 3.0])
        opt = getattr(contrib, method)()
        final_params, _ = _run_opt(opt, fun, init_params, maxiter=50, tol=1e-5)
        self.assertLess(fun(final_params), 1e-5)

    @parameterized.parameters("ssbfgs", "ssbroyden")
    def test_rosenbrock(self, method):
        """Test convergence on the Rosenbrock function."""

        def rosenbrock(x):
            return jnp.sum(
                100.0 * (x[1:] - x[:-1] ** 2) ** 2
                + (1.0 - x[:-1]) ** 2
            )

        init_params = jnp.zeros(2)
        opt = getattr(contrib, method)()
        final_params, _ = _run_opt(
            opt, rosenbrock, init_params,
            maxiter=200, tol=1e-3,
        )
        self.assertLess(rosenbrock(final_params), 1e-3)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            contrib.scale_by_ss_quasi_newton(method="invalid")

    def test_no_linesearch(self):
        """Optimizer works with fixed lr and no linesearch."""
        fun = lambda x: jnp.sum(x**2)
        init_params = jnp.array([1.0, 2.0, 3.0])
        opt = contrib.ssbroyden(learning_rate=0.1, linesearch=None)
        state = opt.init(init_params)
        grad = jax.grad(fun)(init_params)
        updates, state = opt.update(grad, state, init_params)
        new_params = update.apply_updates(init_params, updates)
        # Just verify it runs and produces a different result
        self.assertFalse(jnp.allclose(init_params, new_params))

    def test_state_type(self):
        """Test that the state contains the expected fields."""
        params = jnp.array([1.0, 2.0])
        tx = contrib.scale_by_ss_quasi_newton()
        state = tx.init(params)
        self.assertEqual(state.count, 0)
        self.assertEqual(state.hessian_inv.shape, (2, 2))


if __name__ == "__main__":
    absltest.main()
