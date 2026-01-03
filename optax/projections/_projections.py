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

from collections.abc import Callable
from functools import partial
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
  n_features = values.shape[0]
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


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _projection_unit_sparse_simplex(
    values: jax.typing.ArrayLike,
    max_nz: int,
    use_approx_max_nz: bool = False,
) -> jax.Array:
  """Projection onto the unit sparse simplex."""
  n_features = values.shape[0]
  # Clip max_nz to array size (top_k requires k <= n)
  max_nz = min(max_nz, n_features)
  if use_approx_max_nz:
    top_values, top_indices = jax.lax.approx_max_k(values, max_nz)
  else:
    top_values, top_indices = jax.lax.top_k(values, max_nz)

  cumsum_top = jnp.cumsum(top_values)
  ind = jnp.arange(max_nz) + 1
  cond = 1 / ind + (top_values - cumsum_top / ind) > 0
  idx = jnp.count_nonzero(cond)
  top_projected = jax.nn.relu(
      1 / idx + (top_values - cumsum_top[idx - 1] / idx)
  )
  return (
      jnp.zeros(n_features, dtype=values.dtype)
      .at[top_indices]
      .set(top_projected)
  )


@_projection_unit_sparse_simplex.defjvp
def _projection_unit_sparse_simplex_jvp(
    max_nz: int,
    use_approx_max_nz: bool,
    primals: list[jax.typing.ArrayLike],
    tangents: list[jax.typing.ArrayLike],
) -> tuple[jax.Array, jax.Array]:
  """Custom JVP for sparse simplex projection."""
  (values,) = primals
  (values_dot,) = tangents
  # Note: max_nz clipping happens inside _projection_unit_sparse_simplex
  primal_out = _projection_unit_sparse_simplex(
      values, max_nz, use_approx_max_nz
  )
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
  return primal_out, tangent_out


def projection_sparse_simplex(
    tree: Any,
    max_nz: int,
    scale: jax.typing.ArrayLike = 1,
    use_approx_max_nz: bool = False,
) -> Any:
  r"""Projection onto the sparse simplex.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{s.t.} \quad
    p \ge 0, p^\top 1 = \text{scale}, \|p\|_0 \le \text{max\_nz}

  By default, the projection is onto the sparse probability simplex.

  Args:
    tree: tree to project.
    max_nz: maximum number of non-zero elements allowed in the projection.
    scale: value the projected tree should sum to (default: 1).
    use_approx_max_nz: if True, use approximate top-k selection via
      ``jax.lax.approx_max_k`` which is faster but less accurate (default:
      False).

  Returns:
    projected tree, a tree with the same structure as ``tree``.

  Example:

    Here is an example using an array::

      >>> import jax.numpy as jnp
      >>> from optax import projections
      >>> x = jnp.array([1.0, 1.0, 0.5, 0.5])
      >>> p = projections.projection_sparse_simplex(x, max_nz=2)
      >>> print(jnp.sum(p))
      1.0
      >>> print(jnp.count_nonzero(p))
      2

  References:
    Kyrillidis, A., Becker, S., Cevher, V., & Koch, C.,
    `Sparse Projections onto the Simplex <https://arxiv.org/abs/1206.1529>`_,
    ICML 2013.

  .. versionadded:: 0.2.7
  """
  values, unravel_fn = flatten_util.ravel_pytree(tree)
  new_values = scale * _projection_unit_sparse_simplex(
      values / scale, max_nz=max_nz, use_approx_max_nz=use_approx_max_nz
  )
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
  scalar = (b - optax.tree.vdot(x, a)) / optax.tree.vdot(a, a)
  scalar = jnp.clip(scalar, max=0)
  return optax.tree.add_scale(x, scalar, a)


# ==================== Transport projections ====================
# Based on "Smooth and Sparse Optimal Transport" by Blondel, Seguy, Rolet.
# https://arxiv.org/abs/1710.06276


def _max_l2(
    x: jax.Array, marginal_b: jax.Array, gamma: jax.typing.ArrayLike
) -> jax.Array:
  scale = gamma * marginal_b
  x_scale = x / scale
  p = _projection_unit_simplex(x_scale)
  # From Danskin's theorem, we do not need to backpropagate
  # through projection_simplex.
  p = jax.lax.stop_gradient(p)
  return jnp.dot(x, p) - 0.5 * scale * jnp.dot(p, p)


def _max_ent(
    x: jax.Array, marginal_b: jax.Array, gamma: jax.typing.ArrayLike
) -> jax.Array:
  return gamma * jax.nn.logsumexp(x / gamma) - gamma * jnp.log(marginal_b)


_max_l2_vmap = jax.vmap(_max_l2, in_axes=(1, 0, None))
_max_l2_grad_vmap = jax.vmap(jax.grad(_max_l2), in_axes=(1, 0, None))

_max_ent_vmap = jax.vmap(_max_ent, in_axes=(1, 0, None))
_max_ent_grad_vmap = jax.vmap(jax.grad(_max_ent), in_axes=(1, 0, None))


def _delta_l2(x: jax.Array, gamma: jax.typing.ArrayLike) -> jax.Array:
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with squared l2
  # regularization (see Table 1 in the paper).
  return (0.5 / gamma) * jnp.dot(jax.nn.relu(x), jax.nn.relu(x))


def _delta_ent(x: jax.Array, gamma: jax.typing.ArrayLike) -> jax.Array:
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with negative
  # entropy regularization.
  return gamma * jnp.exp((x / gamma) - 1).sum()


_delta_l2_vmap = jax.vmap(_delta_l2, in_axes=(1, None))
_delta_l2_grad_vmap = jax.vmap(jax.grad(_delta_l2), in_axes=(1, None))

_delta_ent_vmap = jax.vmap(_delta_ent, in_axes=(1, None))
_delta_ent_grad_vmap = jax.vmap(jax.grad(_delta_ent), in_axes=(1, None))


def _make_semi_dual(
    max_vmap: Callable, gamma: jax.typing.ArrayLike = 1.0
) -> Callable:
  """Semi-dual objective, see equation (10) in arxiv.org/abs/1710.06276."""

  def fun(
      alpha: jax.Array,
      cost_matrix: jax.Array,
      marginals_a: jax.Array,
      marginals_b: jax.Array,
  ) -> jax.Array:
    X = alpha[:, jnp.newaxis] - cost_matrix
    ret = jnp.dot(marginals_b, max_vmap(X, marginals_b, gamma))
    ret -= jnp.dot(alpha, marginals_a)
    return ret

  return fun


def _make_dual(delta_vmap: Callable, gamma: jax.typing.ArrayLike) -> Callable:
  """Make dual objective, see equation (7) in arxiv.org/abs/1710.06276."""

  def fun(
      alpha_beta: tuple[jax.Array, jax.Array],
      cost_matrix: jax.Array,
      marginals_a: jax.Array,
      marginals_b: jax.Array,
  ) -> jax.Array:
    alpha, beta = alpha_beta
    alpha_column = alpha[:, jnp.newaxis]
    beta_row = beta[jnp.newaxis, :]
    dual_constraint_matrix = alpha_column + beta_row - cost_matrix
    delta_dual_constraints = delta_vmap(dual_constraint_matrix, gamma)
    dual_loss = (
        delta_dual_constraints.sum()
        - jnp.dot(alpha, marginals_a)
        - jnp.dot(beta, marginals_b)
    )
    return dual_loss

  return fun


def _regularized_transport_semi_dual(
    cost_matrix: jax.Array,
    marginals_a: jax.Array,
    marginals_b: jax.Array,
    make_solver: Callable,
    max_vmap: Callable,
    max_grad_vmap: Callable,
    gamma: jax.typing.ArrayLike = 1.0,
) -> jax.Array:
  """Regularized transport in the semi-dual formulation."""
  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    raise NotImplementedError(
        "Default solver not implemented. Please provide make_solver."
    )

  semi_dual = _make_semi_dual(max_vmap, gamma=gamma)
  solver = make_solver(semi_dual)
  alpha_init = jnp.zeros(size_a)

  alpha = solver.run(
      alpha_init,
      cost_matrix=cost_matrix,
      marginals_a=marginals_a,
      marginals_b=marginals_b,
  ).params

  X = alpha[:, jnp.newaxis] - cost_matrix
  P = max_grad_vmap(X, marginals_b, gamma).T * marginals_b

  return P


def _regularized_transport_dual(
    cost_matrix: jax.Array,
    marginals_a: jax.Array,
    marginals_b: jax.Array,
    make_solver: Callable,
    delta_vmap: Callable,
    delta_grad_vmap: Callable,
    gamma: jax.typing.ArrayLike = 1.0,
) -> jax.Array:
  """Regularized transport in the dual formulation."""
  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    raise NotImplementedError(
        "Default solver not implemented. Please provide make_solver."
    )

  dual = _make_dual(delta_vmap, gamma=gamma)
  solver = make_solver(dual)
  alpha_beta_init = (jnp.zeros(size_a), jnp.zeros(size_b))

  alpha_beta = solver.run(
      init_params=alpha_beta_init,
      cost_matrix=cost_matrix,
      marginals_a=marginals_a,
      marginals_b=marginals_b,
  ).params

  alpha, beta = alpha_beta
  alpha_column = alpha[:, jnp.newaxis]
  beta_row = beta[jnp.newaxis, :]
  dual_constraint_matrix = alpha_column + beta_row - cost_matrix
  plan = delta_grad_vmap(dual_constraint_matrix, gamma).T
  return plan


def projection_transport(
    sim_matrix: jax.Array,
    marginals: tuple[jax.Array, jax.Array],
    make_solver: Callable | None = None,
    use_semi_dual: bool = True,
) -> jax.Array:
  r"""Projection onto the transportation polytope.

  We solve

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ \frac{1}{2}\|S - P\|^2 \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ \langle P, C \rangle
    + \frac{1}{2}\|P\|^2 \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in ref. below)
  using an iterative solver provided via ``make_solver``.

  For a KL-regularized version, see
  :func:`kl_projection_transport <optax.projections.kl_projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: transportation matrix, shape=(size_a, size_b).

  References:
    Blondel, M., Seguy, V., & Rolet, A.,
    `Smooth and Sparse Optimal Transport <https://arxiv.org/abs/1710.06276>`_,
    AISTATS 2018.

  .. versionadded:: 0.2.7
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        max_vmap=_max_l2_vmap,
        max_grad_vmap=_max_l2_grad_vmap,
    )
  else:
    plan = _regularized_transport_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        delta_vmap=_delta_l2_vmap,
        delta_grad_vmap=_delta_l2_grad_vmap,
    )
  return plan


def kl_projection_transport(
    sim_matrix: jax.Array,
    marginals: tuple[jax.Array, jax.Array],
    make_solver: Callable | None = None,
    use_semi_dual: bool = True,
) -> jax.Array:
  r"""Kullback-Leibler projection onto the transportation polytope.

  We solve

  .. math::

    \underset{P > 0}{\text{argmin}} ~ \text{KL}(P, \exp(S)) \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P > 0}{\text{argmin}} ~ \langle P, C \rangle
    + \langle P, \log P \rangle \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in ref. below)
  using an iterative solver provided via ``make_solver``.

  For a squared Euclidean version, see
  :func:`projection_transport <optax.projections.projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: transportation matrix, shape=(size_a, size_b).

  References:
    Blondel, M., Seguy, V., & Rolet, A.,
    `Smooth and Sparse Optimal Transport <https://arxiv.org/abs/1710.06276>`_,
    AISTATS 2018.

  .. versionadded:: 0.2.7
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        max_vmap=_max_ent_vmap,
        max_grad_vmap=_max_ent_grad_vmap,
    )
  else:
    plan = _regularized_transport_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        delta_vmap=_delta_ent_vmap,
        delta_grad_vmap=_delta_ent_grad_vmap,
    )
  return plan


def projection_birkhoff(
    sim_matrix: jax.Array,
    make_solver: Callable | None = None,
    use_semi_dual: bool = True,
) -> jax.Array:
  r"""Projection onto the Birkhoff polytope.

  The Birkhoff polytope is the set of doubly stochastic matrices.
  This is the special case of the transportation polytope with uniform
  marginals.

  We solve

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ \frac{1}{2}\|S - P\|^2 \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = \mathbf{1},
    P \mathbf{1} = \mathbf{1}

  Args:
    sim_matrix: similarity matrix, shape=(n, n). Must be a square matrix.
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: doubly stochastic matrix, shape=(n, n).

  References:
    Blondel, M., Seguy, V., & Rolet, A.,
    `Smooth and Sparse Optimal Transport <https://arxiv.org/abs/1710.06276>`_,
    AISTATS 2018.

  .. versionadded:: 0.2.7
  """
  n = sim_matrix.shape[0]
  marginals = (jnp.ones(n), jnp.ones(n))
  return projection_transport(
      sim_matrix, marginals, make_solver=make_solver,
      use_semi_dual=use_semi_dual,
  )


def kl_projection_birkhoff(
    sim_matrix: jax.Array,
    make_solver: Callable | None = None,
    use_semi_dual: bool = True,
) -> jax.Array:
  r"""Kullback-Leibler projection onto the Birkhoff polytope.

  The Birkhoff polytope is the set of doubly stochastic matrices.
  This is the special case of the transportation polytope with uniform
  marginals.

  We solve

  .. math::

    \underset{P > 0}{\text{argmin}} ~ \text{KL}(P, \exp(S)) \quad
    \textrm{s.t.} \quad P^\top \mathbf{1} = \mathbf{1},
    P \mathbf{1} = \mathbf{1}

  Args:
    sim_matrix: similarity matrix, shape=(n, n). Must be a square matrix.
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: doubly stochastic matrix, shape=(n, n).

  References:
    Blondel, M., Seguy, V., & Rolet, A.,
    `Smooth and Sparse Optimal Transport <https://arxiv.org/abs/1710.06276>`_,
    AISTATS 2018.

  .. versionadded:: 0.2.7
  """
  n = sim_matrix.shape[0]
  marginals = (jnp.ones(n), jnp.ones(n))
  return kl_projection_transport(
      sim_matrix, marginals, make_solver=make_solver,
      use_semi_dual=use_semi_dual,
  )
