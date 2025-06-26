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

"""Euclidean projections."""

from typing import Any

import chex
import jax
from jax import flatten_util
import jax.numpy as jnp
import optax.tree


def projection_non_negative(tree: Any) -> Any:
  r"""Projection onto the non-negative orthant.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  where :math:`x` is the input tree.

  Args:
    tree: tree to project.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return jax.tree.map(jax.nn.relu, tree)


def projection_box(tree: Any, lower: Any, upper: Any) -> Any:
  r"""Projection onto box constraints.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    \text{lower} \le p \le \text{upper}

  where :math:`x` is the input tree.

  Args:
    tree: tree to project.
    lower:  lower bound, a scalar or tree with the same structure as
      ``tree``.
    upper:  upper bound, a scalar or tree with the same structure as
      ``tree``.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return jax.tree.map(jnp.clip, tree, lower, upper)


def projection_hypercube(tree: Any, scale: Any = 1) -> Any:
  r"""Projection onto the (unit) hypercube.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    0 \le p \le \text{scale}

  where :math:`x` is the input tree.

  By default, we project to the unit hypercube (`scale=1`).

  This is a convenience wrapper around
  :func:`projection_box <optax.projections.projection_box>`.

  Args:
    tree: tree to project.
    scale: scale of the hypercube, a scalar or a tree (default: 1).

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return projection_box(tree, lower=0, upper=scale)


@jax.custom_jvp
def _projection_unit_simplex(values: chex.Array) -> chex.Array:
  """Projection onto the unit simplex."""
  s = 1
  n_features = values.shape[0]
  u = jnp.sort(values)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(
    primals: list[chex.Array], tangents: list[chex.Array]
) -> tuple[chex.Array, chex.Array]:
  (values,) = primals
  (values_dot,) = tangents
  primal_out = _projection_unit_simplex(values)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto a simplex.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{scale}

  By default, the projection is onto the probability simplex (unit simplex).

  Args:
    tree: tree to project.
    scale: value the projected tree should sum to (default: 1).

  Returns:
    projected tree, a tree with the same structure as ``tree``.

  Example:

    Here is an example using a tree::

      >>> import jax.numpy as jnp
      >>> from optax import tree, projections
      >>> data = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> print(tree.sum(data))
      6.2
      >>> new_data = projections.projection_simplex(data)
      >>> print(tree.sum(new_data))
      1.0000002

  .. versionadded:: 0.2.3
  """
  values, unravel_fn = flatten_util.ravel_pytree(tree)
  new_values = scale * _projection_unit_simplex(values / scale)
  return unravel_fn(new_values)


def projection_l1_sphere(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l1 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_1 = \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the sphere.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  tree_abs = jax.tree.map(jnp.abs, tree)
  tree_sign = jax.tree.map(jnp.sign, tree)
  tree_abs_proj = projection_simplex(tree_abs, scale)
  return optax.tree.mul(tree_sign, tree_abs_proj)


def projection_l1_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l1 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_1 \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.

  Example:

      >>> import jax.numpy as jnp
      >>> from optax import tree, projections
      >>> data = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> print(tree.norm(data, ord=1))
      6.2
      >>> new_data = projections.projection_l1_ball(data)
      >>> print(tree.norm(new_data, ord=1))
      1.0000002

  .. versionadded:: 0.2.4
  """
  l1_norm = optax.tree.norm(tree, ord=1)
  return jax.lax.cond(
      l1_norm <= scale,
      lambda tree: tree,
      lambda tree: projection_l1_sphere(tree, scale),
      operand=tree,
  )


def projection_l2_sphere(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l2 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_2 = \text{value}

  Args:
    tree: tree to project.
    scale: radius of the sphere.

  Returns:
    projected tree, with the same structure as ``tree``.

  .. versionadded:: 0.2.4
  """
  factor = scale / optax.tree.norm(tree)
  return optax.tree.scale(factor, tree)


def projection_l2_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l2 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_2 \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.

  .. versionadded:: 0.2.4
  """
  squared_norm = optax.tree.norm(tree, squared=True)
  positive = squared_norm > 0
  valid_squared_norm = jnp.where(positive, squared_norm, 1.)
  norm = jnp.where(positive, jnp.sqrt(valid_squared_norm), 0.)
  factor = scale / jnp.maximum(norm, scale)
  return optax.tree.scale(factor, tree)


def projection_linf_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l-infinity ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_{\infty} \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  lower = optax.tree.full_like(tree, -scale)
  upper = optax.tree.full_like(tree, scale)
  return projection_box(tree, lower=lower, upper=upper)


def projection_vector(x: Any, a: Any) -> Any:
  r"""Projection onto a vector.

  Projects a tree ``x`` onto the vector defined by a tree ``a``:

  .. math::

    \operatorname{proj}_a x = \frac{\langle x, a \rangle}{\langle a, a \rangle}
    a

  Args:
    x: tree to project.
    a: tree onto which to project. Must have the same structure as ``x``.

  Returns:
    tree with the same structure as ``x``.
  """
  scalar = optax.tree.vdot(x, a) / optax.tree.vdot(a, a)
  return optax.tree.scale(scalar, a)


def projection_hyperplane(x: Any, a: Any, b: chex.Numeric) -> Any:
  r"""Projection onto a hyperplane.

  Projects a tree ``x`` onto the hyperplane defined by a tree ``a`` and scalar
  ``b``.

  .. math::

    \operatorname{argmin}_y \|x - y\|_2^2 \quad \text{subject to} \quad
    \langle a, y \rangle = b

  Args:
    x: tree to project.
    a: tree defining hyperplane onto which to project. Must have the same
      structure as ``x``.
    b: scalar defining hyperplane onto which to project.

  Returns:
    tree with the same structure as ``x``.
  """
  scalar = (b - optax.tree.vdot(x, a)) / optax.tree.vdot(a, a)
  return optax.tree.add_scale(x, scalar, a)


def projection_halfspace(x: Any, a: Any, b: chex.Numeric) -> Any:
  r"""Projection onto a halfspace.

  Projects a tree ``x`` onto the halfspace defined by a tree ``a`` and scalar
  ``b``.

  .. math::

    \operatorname{argmin}_y \|x - y\|_2^2 \quad \text{subject to} \quad
    \langle a, y \rangle \leq b

  Args:
    x: tree to project.
    a: tree defining halfspace onto which to project. Must have the same
      structure as ``x``.
    b: scalar defining halfspace onto which to project.

  Returns:
    tree with the same structure as ``x``.
  """
  scalar = (b - optax.tree.vdot(x, a)) / optax.tree.vdot(a, a)
  scalar = scalar.clip(max=0)
  return optax.tree.add_scale(x, scalar, a)
