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
def _projection_unit_simplex(values: jax.typing.ArrayLike) -> jax.Array:
  """Projection onto the unit simplex."""
  s = 1
  # pyrefly: ignore [missing-attribute]
  n_features = values.shape[0]  # pytype: disable=attribute-error  # jax-arraylike # noqa: E501
  u = jnp.sort(values)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(
    primals: list[jax.typing.ArrayLike], tangents: list[jax.typing.ArrayLike]
) -> tuple[jax.Array, jax.Array]:
  (values,) = primals
  (values_dot,) = tangents
  primal_out = _projection_unit_simplex(values)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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


def projection_l1_sphere(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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


def projection_l1_ball(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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


def projection_l2_sphere(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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


def projection_l2_ball(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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
  factor = scale / jnp.sqrt(jnp.maximum(squared_norm, scale**2))
  return optax.tree.scale(factor, tree)


def projection_linf_ball(tree: Any, scale: jax.typing.ArrayLike = 1) -> Any:
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
  # pyrefly: ignore[unsupported-operation]
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


def projection_hyperplane(x: Any, a: Any, b: jax.typing.ArrayLike) -> Any:
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
  # pyrefly: ignore[unsupported-operation]
  scalar = (b - optax.tree.vdot(x, a)) / optax.tree.vdot(a, a)
  return optax.tree.add_scale(x, scalar, a)


def projection_halfspace(x: Any, a: Any, b: jax.typing.ArrayLike) -> Any:
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
  # pyrefly: ignore[unsupported-operation]
  scalar = (b - optax.tree.vdot(x, a)) / optax.tree.vdot(a, a)
  scalar = jnp.clip(scalar, max=0)
  return optax.tree.add_scale(x, scalar, a)


def projection_affine_set(
    x: jax.typing.ArrayLike,
    a: jax.typing.ArrayLike,
    b: jax.typing.ArrayLike,
) -> jax.Array:
  r"""Projection onto an affine set.

  Projects a vector ``x`` onto the affine set defined by a matrix ``a`` and a
  vector ``b``.

  .. math::

    \operatorname{argmin}_y \|x - y\|_2^2 \quad \text{subject to} \quad
    a y = b

  The projection is computed in closed form,
  :math:`y = x + a^\top (a a^\top)^{-1} (b - a x)`.

  Args:
    x: array of shape ``(n,)`` to project.
    a: matrix of shape ``(m, n)``, with ``m <= n``, defining the affine set.
      Must have linearly independent rows.
    b: vector of shape ``(m,)`` defining the affine set.

  Returns:
    projected array, with the same shape as ``x``.

  Example:

    >>> import jax.numpy as jnp
    >>> from optax import projections
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> a = jnp.array([[1.0, 1.0, 1.0]])
    >>> b = jnp.array([3.0])
    >>> print(projections.projection_affine_set(x, a, b))
    [0. 1. 2.]

  .. versionadded:: 0.2.9
  """
  x = jnp.asarray(x)
  a = jnp.asarray(a)
  b = jnp.asarray(b)
  return x + a.T @ jnp.linalg.solve(a @ a.T, b - a @ x)


def projection_box_section(
    x: jax.typing.ArrayLike,
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    w: jax.typing.ArrayLike,
    c: jax.typing.ArrayLike,
) -> jax.Array:
  r"""Projection onto a section of a box.

  Projects a vector ``x`` onto the intersection of a box (hyperrectangle)
  and a hyperplane with positive coefficient vector ``w``.

  .. math::

    \operatorname{argmin}_y \|x - y\|_2^2 \quad \text{subject to} \quad
    \text{lower} \le y \le \text{upper}, \langle w, y \rangle = c

  The solution has the form :math:`y_i = \operatorname{clip}(x_i + \tau w_i,
  \text{lower}_i, \text{upper}_i)`, where the scalar :math:`\tau` is the root
  of a monotone function, found by bisection. The projection is
  differentiable, via implicit differentiation of the root.

  The constraint set is non-empty if and only if :math:`\langle w,
  \text{lower} \rangle \le c \le \langle w, \text{upper} \rangle`; the result
  is undefined otherwise.

  Args:
    x: array of shape ``(n,)`` to project.
    lower: lower bound of the box, a scalar or an array broadcastable to the
      shape of ``x``.
    upper: upper bound of the box, a scalar or an array broadcastable to the
      shape of ``x``.
    w: weights of the hyperplane, an array with the same shape as ``x``. All
      entries must be positive.
    c: scalar defining the hyperplane.

  Returns:
    projected array, with the same shape as ``x``.

  Example:

    >>> import jax.numpy as jnp
    >>> from optax import projections
    >>> x = jnp.array([0.5, 1.5])
    >>> w = jnp.array([1.0, 1.0])
    >>> print(projections.projection_box_section(x, 0.0, 1.0, w, 1.0))
    [0. 1.]

  .. versionadded:: 0.2.9
  """
  x = jnp.asarray(x)
  w = jnp.asarray(w)

  def residual(tau):
    # Monotonically non-decreasing in tau, since w > 0.
    return jnp.dot(w, jnp.clip(x + tau * w, lower, upper)) - c

  # For tau below (resp. above) the bracket, all coordinates of the candidate
  # solution hit the lower (resp. upper) bound, so by feasibility the residual
  # is non-positive (resp. non-negative) and the bracket contains a root.
  bracket_low = jax.lax.stop_gradient(jnp.min((lower - x) / w))
  bracket_high = jax.lax.stop_gradient(jnp.max((upper - x) / w))

  def bisect(fun, init):
    del init  # the root is bracketed by construction

    def body_fun(_, bracket):
      low, high = bracket
      mid = 0.5 * (low + high)
      go_left = fun(mid) >= 0
      return jnp.where(go_left, low, mid), jnp.where(go_left, mid, high)

    # 100 iterations narrow the bracket by a factor of 2**100, well below
    # float64 resolution for any realistic input.
    low, high = jax.lax.fori_loop(0, 100, body_fun, (bracket_low, bracket_high))
    return 0.5 * (low + high)

  def tangent_solve(g, y):
    return y / g(1.0)

  tau = jax.lax.custom_root(residual, 0.0, bisect, tangent_solve)
  return jnp.clip(x + tau * w, lower, upper)
