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
"""The restarted Halpern primal-dual hybrid gradient method."""

from jax import lax, numpy as jnp
from optax import tree_utils as otu


def solve_canonical(
  c, A, b, iters, reflect=True, restarts=True, tau=None, sigma=None
):
  r"""Solves a linear program using the restarted Halpern primal-dual hybrid
  gradient (RHPDHG) method.

  Minimizes :math:`c \cdot x` subject to :math:`A x = b` and :math:`x \geq 0`.

  See also `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_.

  Args:
    c: Cost vector.
    A: Equality constraint matrix.
    b: Equality constraint vector.
    iters: Number of iterations to run the solver for.
    reflect: Use reflection. See paper for details.
    restarts: Use restarts. See paper for details.
    tau: Primal step size. See paper for details.
    sigma: Dual step size. See paper for details.

  Returns:
    A dictionary whose entries are as follows:
      - primal: The final primal solution.
      - dual: The final dual solution.
      - primal_iterates: The primal iterates.
      - dual_iterates: The dual iterates.

  Examples:
    >>> from jax import numpy as jnp
    >>> import optax
    >>> c = -jnp.array([2, 1])
    >>> A = jnp.zeros([0, 2])
    >>> b = jnp.zeros(0)
    >>> G = jnp.array([[3, 1], [1, 1], [1, 4]])
    >>> h = jnp.array([21, 9, 24])
    >>> x = optax.linprog.rhpdhg(c, A, b, G, h, 1_000_000)['primal']
    >>> print(x[0])
    5.99...
    >>> print(x[1])
    2.99...

  References:
    Haihao Lu, Jinwen Yang, `Restarted Halpern PDHG for Linear Programming
    <https://arxiv.org/abs/2407.16144>`_, 2024
    Haihao Lu, Zedong Peng, Jinwen Yang, `MPAX: Mathematical Programming in JAX
    <https://arxiv.org/abs/2412.09734>`_, 2024
  """

  if tau is None or sigma is None:
    A_norm = jnp.linalg.norm(A, axis=(0, 1), ord=2)
    if tau is None:
      tau = 1 / (2 * A_norm)
    if sigma is None:
      sigma = 1 / (2 * A_norm)

  def T(z):
    # primal dual hybrid gradient (PDHG)
    x, y = z
    xn = x + tau * (y @ A - c)
    xn = xn.clip(min=0)
    yn = y + sigma * (b - A @ (2 * xn - x))
    return xn, yn

  def H(z, k, z0):
    # Halpern PDHG
    Tz = T(z)
    if reflect:
      zc = otu.tree_sub(otu.tree_scalar_mul(2, Tz), z)
    else:
      zc = Tz
    kp2 = k + 2
    zn = otu.tree_add(
      otu.tree_scalar_mul((k + 1) / kp2, zc),
      otu.tree_scalar_mul(1 / kp2, z0),
    )
    return zn, Tz

  def update(carry, _):
    z, k, z0, d0 = carry
    zn, Tz = H(z, k, z0)

    if restarts:
      d = otu.tree_l2_norm(otu.tree_sub(z, Tz), squared=True)
      restart = d <= d0 * jnp.exp(-2)
      new_carry = otu.tree_where(
        restart,
        (zn, 0, zn, d),
        (zn, k + 1, z0, d0),
      )
    else:
      new_carry = zn, k + 1, z0, d0

    return new_carry, z

  def run():
    m, n = A.shape
    x = jnp.zeros(n)
    y = jnp.zeros(m)
    z0 = x, y
    d0 = otu.tree_l2_norm(otu.tree_sub(z0, T(z0)), squared=True)
    (z, _, _, _), zs = lax.scan(update, (z0, 0, z0, d0), length=iters)
    x, y = z
    xs, ys = zs
    return {
      "primal": x,
      "dual": y,
      "primal_iterates": xs,
      "dual_iterates": ys,
    }

  return run()


def general_to_canonical(c, A, b, G, h):
  """Converts a linear program from general form to canonical form.

  The solution to the new linear program will consist of the concatenation of
    - the positive part of x
    - the negative part of x
    - slacks

  That is, we go from

    Minimize c · x subject to
      A x = b
      G x ≤ h

  to

    Minimize c · (x⁺ - x⁻) subject to
      A (x⁺ - x⁻) = b
      G (x⁺ - x⁻) + s = h
      x⁺, x⁻, s ≥ 0

  Args:
    c: Cost vector.
    A: Equality constraint matrix.
    b: Equality constraint vector.
    G: Inequality constraint matrix.
    h: Inequality constraint vector.

  Returns:
    A triple (c', A', b') representing the corresponding canonical form.
  """
  c_can = jnp.concatenate([c, -c, jnp.zeros(h.size)])
  G_ = jnp.concatenate([G, -G, jnp.eye(h.size)], 1)
  A_ = jnp.concatenate([A, -A, jnp.zeros([b.size, h.size])], 1)
  A_can = jnp.concatenate([A_, G_], 0)
  b_can = jnp.concatenate([b, h])
  return c_can, A_can, b_can


def solve_general(
  c, A, b, G, h, iters, reflect=True, restarts=True, tau=None, sigma=None
):
  r"""Solves a linear program using the restarted Halpern primal-dual hybrid
  gradient (RHPDHG) method.

  Minimizes :math:`c \cdot x` subject to :math:`A x = b` and :math:`G x \leq h`.

  See also `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_.

  Args:
    c: Cost vector.
    A: Equality constraint matrix.
    b: Equality constraint vector.
    G: Inequality constraint matrix.
    h: Inequality constraint vector.
    iters: Number of iterations to run the solver for.
    reflect: Use reflection. See paper for details.
    restarts: Use restarts. See paper for details.
    tau: Primal step size. See paper for details.
    sigma: Dual step size. See paper for details.

  Returns:
    A dictionary whose entries are as follows:
      - primal: The final primal solution.
      - slacks: The final primal slack values.
      - canonical_result: The result for the canonical program that was used
        internally to find this solution. See paper for details.

  References:
    Haihao Lu, Jinwen Yang, `Restarted Halpern PDHG for Linear Programming
    <https://arxiv.org/abs/2407.16144>`_, 2024
    Haihao Lu, Zedong Peng, Jinwen Yang, `MPAX: Mathematical Programming in JAX
    <https://arxiv.org/abs/2412.09734>`_, 2024
  """
  canonical = general_to_canonical(c, A, b, G, h)
  result = solve_canonical(*canonical, iters, reflect, restarts, tau, sigma)
  x_pos, x_neg, slacks = jnp.split(result["primal"], [c.size, c.size * 2])
  return {
    "primal": x_pos - x_neg,
    "slacks": slacks,
    "canonical_result": result,
  }
