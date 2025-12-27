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
from typing import Any, Optional

import jax
import jax.numpy as jnp
from optax._src import numerics


def tree_add(tree_x: Any, tree_y: Any, *other_trees: Any) -> Any:
  r"""Add two (or more) pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
    *other_trees: optional other trees to add

  Returns:
    the sum of the two (or more) pytrees.

  .. versionchanged:: 0.2.1
    Added optional ``*other_trees`` argument.
  """
  trees = [tree_x, tree_y, *other_trees]
  return jax.tree.map(lambda *leaves: sum(leaves), *trees)


def tree_sub(tree_x: Any, tree_y: Any) -> Any:
  r"""Subtract two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the difference of the two pytrees.
  """
  return jax.tree.map(operator.sub, tree_x, tree_y)


def tree_mul(tree_x: Any, tree_y: Any) -> Any:
  r"""Multiply two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the product of the two pytrees.
  """
  return jax.tree.map(operator.mul, tree_x, tree_y)


def tree_div(tree_x: Any, tree_y: Any) -> Any:
  r"""Divide two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the quotient of the two pytrees.
  """
  return jax.tree.map(operator.truediv, tree_x, tree_y)


def tree_scale(
    scalar: jax.typing.ArrayLike,
    tree: Any,
) -> Any:
  r"""Multiply a tree by a scalar.

  In infix notation, the function performs ``out = scalar * tree``.

  Args:
    scalar: scalar value.
    tree: pytree.

  Returns:
    a pytree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: scalar * x, tree)


def tree_add_scale(
    tree_x: Any, scalar: jax.typing.ArrayLike, tree_y: Any
) -> Any:
  r"""Add two trees, where the second tree is scaled by a scalar.

  In infix notation, the function performs ``out = tree_x + scalar * tree_y``.

  Args:
    tree_x: first pytree.
    scalar: scalar value.
    tree_y: second pytree.

  Returns:
    a pytree with the same structure as ``tree_x`` and ``tree_y``.
  """
  scalar = jnp.asarray(scalar)
  return jax.tree.map(
      lambda x, y: (None if x is None else (x + scalar * y)),
      tree_x, tree_y, is_leaf=lambda x: x is None)


def _vdot(a, b, precision=jax.lax.Precision.HIGHEST):
  """Compute the inner product between two (possibly complex) arrays."""
  if jax.__version__ < "0.7.2":
    return jnp.vdot(a, b, precision=precision)
  assert a.shape == b.shape
  if jax.dtypes.issubdtype(a.dtype, jnp.complexfloating):
    a = jnp.conj(a)
  # jnp.vdot internally uses ravel(), which leads to undesirable comms
  # in distributed setups, and failures when using explicit-sharding.
  mesh = jax.typeof(a).sharding.mesh
  if mesh.are_all_axes_explicit:
    sharding = jax.sharding.NamedSharding(mesh, jax.P())
  else:
    sharding = None
  return jnp.tensordot(a, b, a.ndim, precision=precision, out_sharding=sharding)


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x: Any, tree_y: Any) -> jax.typing.ArrayLike:
  r"""Compute the inner product between two pytrees.

  Args:
    tree_x: first pytree to use.
    tree_y: second pytree to use.

  Returns:
    inner product between ``tree_x`` and ``tree_y``, a scalar value.

  Examples:

    >>> optax.tree_utils.tree_vdot(
    ...   {'a': jnp.array([1, 2]), 'b': jnp.array([1, 2])},
    ...   {'a': jnp.array([-1, -1]), 'b': jnp.array([1, 1])},
    ... )
    Array(0, dtype=int32)

  .. note::
    We upcast the values to the highest precision to avoid
    numerical issues.
  """
  vdots = jax.tree.map(_vdot_safe, tree_x, tree_y)
  return jax.tree.reduce(operator.add, vdots, initializer=0)


def tree_sum(tree: Any) -> jax.typing.ArrayLike:
  """Compute the sum of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  sums = jax.tree.map(jnp.sum, tree)
  return jax.tree.reduce(operator.add, sums, initializer=0)


def tree_max(tree: Any) -> jax.typing.ArrayLike:
  """Compute the max of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  def f(array):
    if jnp.size(array) == 0:
      return None
    else:
      return jnp.max(array)
  maxes = jax.tree.map(f, tree)
  return jax.tree.reduce(jnp.maximum, maxes, initializer=-float("inf"))


def tree_min(tree: Any) -> jax.typing.ArrayLike:
  """Compute the min of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  def f(array):
    if jnp.size(array) == 0:
      return None
    else:
      return jnp.min(array)
  mins = jax.tree.map(f, tree)
  return jax.tree.reduce(jnp.minimum, mins, initializer=float("inf"))


def tree_size(tree: Any) -> int:
  r"""Total size of a pytree.

  Args:
    tree: pytree

  Returns:
    the total size of the pytree.
  """
  return sum(jnp.size(leaf) for leaf in jax.tree.leaves(tree))


def tree_conj(tree: Any) -> Any:
  """Compute the conjugate of a pytree.

  Args:
    tree: pytree.

  Returns:
    a pytree with the same structure as ``tree``.
  """
  return jax.tree.map(jnp.conj, tree)


def tree_real(tree: Any) -> Any:
  """Compute the real part of a pytree.

  Args:
    tree: pytree.

  Returns:
    a pytree with the same structure as ``tree``.
  """
  return jax.tree.map(jnp.real, tree)


def _square(leaf):
  return jnp.square(leaf.real) + jnp.square(leaf.imag)


def tree_norm(tree: Any,
              ord: int | str | float | None = None,  # pylint: disable=redefined-builtin
              squared: bool = False) -> jax.Array:
  """Compute the vector norm of the given ord of a pytree.

  Args:
    tree: pytree.
    ord: the order of the vector norm to compute from (None, 1, 2, inf).
    squared: whether the norm should be returned squared or not.

  Returns:
    a scalar value.
  """
  if ord is None or ord == 2:
    squared_tree = jax.tree.map(_square, tree)
    sqnorm = tree_sum(squared_tree)
    return jnp.array(sqnorm if squared else jnp.sqrt(sqnorm))
  elif ord == 1:
    ret = tree_sum(jax.tree.map(jnp.abs, tree))
  elif ord == jnp.inf or ord in ("inf", "infinity"):
    ret = tree_max(jax.tree.map(jnp.abs, tree))
  else:
    raise ValueError(f"Unsupported ord: {ord}")
  return jnp.array(ret if not squared else _square(ret))


def tree_batch_shape(
    tree: Any,
    shape: tuple[int, ...] = (),
):
  """Add leading batch dimensions to each leaf of a pytree.

  Args:
    tree: a pytree.
    shape: a shape indicating what leading batch dimensions to add.

  Returns:
    a pytree with the leading batch dimensions added.
  """
  return jax.tree.map(
      lambda x: jnp.broadcast_to(x, (*shape, *jnp.shape(x))), tree
  )


def tree_zeros_like(
    tree: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an all-zeros tree with the same structure.

  Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of zeros.

  Returns:
    an all-zeros tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def tree_ones_like(
    tree: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an all-ones tree with the same structure.

  Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of ones.

  Returns:
    an all-ones tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.ones_like(x, dtype=dtype), tree)


def tree_full_like(
    tree: Any,
    fill_value: jax.typing.ArrayLike,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an identical tree where all tensors are filled with ``fill_value``.

  Args:
    tree: pytree.
    fill_value: the fill value for all tensors in the tree.
    dtype: optional dtype to use for the tensors in the tree.

  Returns:
    an tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.full_like(x, fill_value, dtype=dtype), tree)


def tree_clip(
    tree: Any,
    min_value: Optional[jax.typing.ArrayLike] = None,
    max_value: Optional[jax.typing.ArrayLike] = None,
) -> Any:
  """Creates an identical tree where all tensors are clipped to `[min, max]`.

  Args:
    tree: pytree.
    min_value: optional minimal value to clip all tensors to. If ``None``
      (default) then result will not be clipped to any minimum value.
    max_value: optional maximal value to clip all tensors to. If ``None``
      (default) then result will not be clipped to any maximum value.

  Returns:
    a tree with the same structure as ``tree``.

  .. versionadded:: 0.2.3
  """
  return jax.tree.map(lambda g: jnp.clip(g, min_value, max_value), tree)


def tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree.map(
      lambda g, t: (
          (1 - decay) * (g**order) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


def tree_update_infinity_moment(updates, moments, decay, eps):
  """Compute the exponential moving average of the infinity norm."""
  return jax.tree.map(
      lambda g, t: (
          jnp.maximum(jnp.abs(g) + eps, decay * t) if g is not None else g
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


def tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order

    half_order = order / 2
    # JAX generates different HLO for int and float `order`
    if half_order.is_integer():
      half_order = int(half_order)
    return numerics.abs_sq(g) ** half_order

  return jax.tree.map(
      lambda g, t: (
          (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


@functools.partial(jax.jit, inline=True)
def tree_bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree.map(lambda t: t / bias_correction_.astype(t.dtype), moment)


def tree_where(condition, tree_x, tree_y):
  """Select tree_x values if condition is true else tree_y values.

  Args:
    condition: boolean specifying which values to select from tree x or tree_y
    tree_x: pytree chosen if condition is True
    tree_y: pytree chosen if condition is False

  Returns:
    tree_x or tree_y depending on condition.
  """
  return jax.tree.map(lambda x, y: jnp.where(condition, x, y), tree_x, tree_y)


def tree_allclose(
    a: Any,
    b: Any,
    rtol: jax.typing.ArrayLike = 1e-05,
    atol: jax.typing.ArrayLike = 1e-08,
    equal_nan: bool = False
):
  """Check whether two trees are element-wise approximately equal within a tolerance.

  See :func:`jax.numpy.allclose` for the equivalent on arrays.

  Args:
    a: a tree
    b: a tree
    rtol: relative tolerance used for approximate equality
    atol: absolute tolerance used for approximate equality
    equal_nan: boolean indicating whether NaNs are treated as equal

  Returns:
    a boolean value.
  """  # noqa: E501
  def f(a, b):
    return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
  tree = jax.tree.map(f, a, b)
  leaves = jax.tree.leaves(tree)
  result = functools.reduce(operator.and_, leaves, True)
  return result
