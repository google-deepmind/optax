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

"""Utilities to perform maths on pytrees."""

import functools
import operator
from typing import Any

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp


def tree_add(tree_x: Any, tree_y: Any) -> Any:
  r"""Add two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
  Returns:
    the sum of the two pytrees.
  """
  return jtu.tree_map(operator.add, tree_x, tree_y)


def tree_sub(tree_x: Any, tree_y: Any) -> Any:
  r"""Subtract two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
  Returns:
    the difference of the two pytrees.
  """
  return jtu.tree_map(operator.sub, tree_x, tree_y)


def tree_mul(tree_x: Any, tree_y: Any) -> Any:
  r"""Multiply two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
  Returns:
    the product of the two pytrees.
  """
  return jtu.tree_map(operator.mul, tree_x, tree_y)


def tree_div(tree_x: Any, tree_y: Any) -> Any:
  r"""Divide two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
  Returns:
    the quotient of the two pytrees.
  """
  return jtu.tree_map(operator.truediv, tree_x, tree_y)


def tree_scalar_mul(scalar: float, tree: Any) -> Any:
  r"""Multiply a tree by a scalar.

  In infix notation, the function performs ``out = scalar * tree``.

  Args:
    scalar: scalar value.
    tree: pytree.
  Returns:
    a pytree with the same structure as ``tree``.
  """
  return jtu.tree_map(lambda x: scalar * x, tree)


def tree_add_scalar_mul(tree_x: Any, scalar: float, tree_y: Any) -> Any:
  r"""Add two trees, where the second tree is scaled by a scalar.

  In infix notation, the function performs ``out = tree_x + scalar * tree_y``.

  Args:
    tree_x: first pytree.
    scalar: scalar value.
    tree_y: second pytree.
  Returns:
    a pytree with the same structure as ``tree_x`` and ``tree_y``.
  """
  return jtu.tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x: Any, tree_y: Any) -> chex.Numeric:
  r"""Compute the inner product between two pytrees.

  Args:
    tree_x: first pytree to use.
    tree_y: second pytree to use.
  Returns:
    inner product between ``tree_x`` and ``tree_y``, a scalar value.

  >>> optax.tree_utils.tree_vdot(
  >>>   {a: jnp.array([1, 2]), b: jnp.array([1, 2])},
  >>>   {a: jnp.array([-1, -1]), b: jnp.array([1, 1])},
  >>> )
  0.0

  Implementation detail: we upcast the values to the highest precision to avoid
  numerical issues.
  """
  vdots = jtu.tree_map(_vdot_safe, tree_x, tree_y)
  return jtu.tree_reduce(operator.add, vdots)


def tree_sum(tree: Any) -> chex.Numeric:
  """Compute the sum of all the elements in a pytree.

  Args:
    tree: pytree.
  Returns:
    a scalar value.
  """
  sums = jtu.tree_map(jnp.sum, tree)
  return jtu.tree_reduce(operator.add, sums)


def _square(leaf):
  return jnp.square(leaf.real) + jnp.square(leaf.imag)


def tree_l2_norm(tree: Any, squared: bool = False) -> chex.Numeric:
  """Compute the l2 norm of a pytree.

  Args:
    tree: pytree.
    squared: whether the norm should be returned squared or not.
  Returns:
    a scalar value.
  """
  squared_tree = jtu.tree_map(_square, tree)
  sqnorm = tree_sum(squared_tree)
  if squared:
    return sqnorm
  else:
    return jnp.sqrt(sqnorm)


def tree_zeros_like(tree: Any) -> Any:
  """Creates an all-zeros tree with the same structure.

  Args:
    tree: pytree.
  Returns:
    an all-zeros tree with the same structure as ``tree``.
  """
  return jtu.tree_map(jnp.zeros_like, tree)


def tree_ones_like(tree: Any) -> Any:
  """Creates an all-ones tree with the same structure.

  Args:
    tree: pytree.
  Returns:
    an all-ones tree with the same structure as ``tree``.
  """
  return jtu.tree_map(jnp.ones_like, tree)
