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
"""Tests for the restarted Halpern primal-dual hybrid gradient method."""

from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import cvxpy as cp

from optax.linprog import rhpdhg


def solve_cvxpy(c, A, b, G, h):
  x = cp.Variable(c.size)
  constraints = []
  if A.shape[0] > 0:
    constraints.append(A @ x == b)
  if G.shape[0] > 0:
    constraints.append(G @ x <= h)
  objective = cp.Minimize(c @ x)
  problem = cp.Problem(objective, constraints)
  problem.solve(solver='GLPK')
  return x.value, problem.status


class RHPDHGTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.f = jax.jit(partial(rhpdhg, iters=100_000))

  @parameterized.parameters(
    dict(n_vars=n_vars, n_eq=n_eq, n_ineq=n_ineq)
    for n_vars in range(8)
    for n_eq in range(n_vars)
    for n_ineq in range(8)
    if n_eq + n_ineq >= n_vars
    # Make sure set of solvable LPs with these shapes is not null in measure.
  )
  def test_hungarian_algorithm(self, n_vars, n_eq, n_ineq):
    # Find a solvable LP.
    while True:

      c = np.random.normal(size=(n_vars,))
      A = np.random.normal(size=(n_eq, n_vars))
      b = np.random.normal(size=(n_eq,))
      G = np.random.normal(size=(n_ineq, n_vars))
      h = np.random.normal(size=(n_ineq,))

      # For numerical testing purposes, constrain x to [-limit, limit].
      limit = 5
      G = jnp.concatenate([G, jnp.eye(n_vars), -jnp.eye(n_vars)])
      h = jnp.concatenate([h, jnp.full(n_vars * 2, limit)])

      r, status = solve_cvxpy(c, A, b, G, h)

      if status == 'optimal':
        break

    result = self.f(c, A, b, G, h)
    x = result['primal']

    rtol = 1e-2
    atol = 1e-2

    with self.subTest('x approximately satisfies equality constraints'):
      np.testing.assert_allclose(A @ x, b, rtol=rtol, atol=atol)

    with self.subTest('x approximately satisfies inequality constraints'):
      np.testing.assert_allclose((G @ x).clip(min=h), h, rtol=rtol, atol=atol)

    with self.subTest('x is approximately as good as the reference solution'):
      cx = c @ x
      cr = c @ r
      np.testing.assert_allclose(cx.clip(min=cr), cr, rtol=rtol, atol=atol)


if __name__ == '__main__':
  absltest.main()
