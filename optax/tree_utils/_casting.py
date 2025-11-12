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
"""Utilities to cast pytrees to specific dtypes."""

import functools
from typing import Optional, TypeVar

import chex
import jax
import jax.numpy as jnp

T = TypeVar('T')


def tree_cast_like(tree: T, other_tree: chex.ArrayTree) -> T:
  """Cast tree to dtypes of other_tree.

  Args:
    tree: the tree to cast.
    other_tree: reference array tree to use to cast to dtypes of leaves

  Returns:
    the tree, with leaves cast to dtype.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
    ...         'c': jnp.array(2.0, dtype=jnp.float32)}
    >>> other_tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
    ...               'c': jnp.array(2.0, dtype=jnp.bfloat16)}
    >>> optax.tree_utils.tree_cast_like(tree, other_tree)
    {'a': {'b': Array(1., dtype=float32)}, 'c': Array(2, dtype=bfloat16)}
  """
  return jax.tree.map(lambda t, o: t.astype(o.dtype)
                      if hasattr(o, 'dtype') else t, tree, other_tree)


def tree_cast(
    tree: chex.ArrayTree, dtype: Optional[jax.typing.DTypeLike]
) -> chex.ArrayTree:
  """Cast tree to given dtype, skip if None.

  Args:
    tree: the tree to cast.
    dtype: the dtype to cast to, or None to skip.

  Returns:
    the tree, with leaves cast to dtype.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
    ...         'c': jnp.array(2.0, dtype=jnp.float32)}
    >>> optax.tree_utils.tree_cast(tree, dtype=jnp.bfloat16)
    {'a': {'b': Array(1, dtype=bfloat16)}, 'c': Array(2, dtype=bfloat16)}
  """
  if dtype is not None:
    return jax.tree.map(lambda t: t.astype(dtype), tree)
  else:
    return tree


def tree_dtype(
    tree: chex.ArrayTree, mixed_dtype_handler: Optional[str] = None
) -> jax.typing.DTypeLike:
  """Fetch dtype of tree.

  If the tree is empty, returns the default dtype of JAX arrays.

  Args:
    tree: the tree to fetch the dtype of.
    mixed_dtype_handler: how to handle mixed dtypes in the tree.  - If
      ``mixed_dtype_handler=None``, returns the common dtype of the leaves of
      the tree if it exists, otherwise raises an error. - If
      ``mixed_dtype_handler='promote'``, promotes the dtypes of the leaves of
      the tree to a common promoted dtype using :func:`jax.numpy.promote_types`.
      - If ``mixed_dtype_handler='highest'`` or
      ``mixed_dtype_handler='lowest'``, returns the highest/lowest dtype of the
      leaves of the tree. We consider a partial ordering of dtypes as ``dtype1
      <= dtype2`` if ``dtype1`` is promoted to ``dtype2``, that is, if
      ``jax.numpy.promote_types(dtype1, dtype2) == dtype2``. Since some dtypes
      cannot be promoted to one another, this is not a total ordering, and the
      'highest' or 'lowest' options may not be applicable. These options will
      throw an error if the dtypes of the leaves of the tree cannot be promoted
      to one another.

  Returns:
    the dtype of the tree.

  Raises:
    ValueError: If ``mixed_dtype_handler`` is set to ``None`` and multiple
      dtypes are found in the tree.
    ValueError: If ``mixed_dtype_handler`` is set to  ``'highest'`` or
      ``'lowest'`` and some leaves' dtypes in the tree cannot be promoted to one
      another.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
    ...         'c': jnp.array(2.0, dtype=jnp.float32)}
    >>> optax.tree_utils.tree_dtype(tree)
    dtype('float32')
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float16)},
    ...         'c': jnp.array(2.0, dtype=jnp.float32)}
    >>> optax.tree_utils.tree_dtype(tree, 'lowest')
    dtype('float16')
    >>> optax.tree_utils.tree_dtype(tree, 'highest')
    dtype('float32')
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.int32)},
    ...         'c': jnp.array(2.0, dtype=jnp.uint32)}
    >>> # optax.tree_utils.tree_dtype(tree, 'highest')
    >>> # -> will throw an error because int32 and uint32
    >>> # cannot be promoted to one another.
    >>> optax.tree_utils.tree_dtype(tree, 'promote')
    dtype('int32')

  .. seealso:: :func:`jax.numpy.promote_types`,
    `Type promotion semantics in JAX
    <https://jax.readthedocs.io/en/latest/type_promotion.html#type-promotion>`_

  .. versionadded:: 0.2.4
  """
  leaves = jax.tree.leaves(tree)
  if not leaves:
    # If the tree is empty, we return the default dtype as given by JAX on
    # empty lists.
    return jnp.dtype(jnp.asarray(leaves))
  if mixed_dtype_handler is None:
    dtype = jnp.asarray(leaves[0]).dtype
    _tree_assert_all_dtypes_equal(tree, dtype)
    return dtype
  if mixed_dtype_handler == 'promote':
    promoted_dtype = functools.reduce(
        jnp.promote_types, [jnp.asarray(x).dtype for x in leaves]
    )
    return promoted_dtype
  if mixed_dtype_handler == 'highest':
    highest_dtype = functools.reduce(
        _higher_dtype, [jnp.asarray(x).dtype for x in leaves]
    )
    return highest_dtype
  if mixed_dtype_handler == 'lowest':
    lowest_dtype = functools.reduce(
        _lower_dtype, [jnp.asarray(x).dtype for x in leaves]
    )
    return lowest_dtype
  raise ValueError(
      f'Invalid value for {mixed_dtype_handler=}, possible values are: None,'
      ' "promote", "highest", "lowest".'
  )


def _tree_assert_all_dtypes_equal(
    tree: chex.ArrayTree, dtype: jax.typing.DTypeLike
) -> None:
  """Checks that all leaves of the tree have the given dtype.

  Args:
    tree: the tree to check.
    dtype: the dtype to check against.

  Raises:
    ValueError: If any element of the tree does not match the given dtype.
  """

  def _assert_dtypes_equal(path, x):
    x_dtype = jnp.asarray(x).dtype
    if x_dtype != dtype:
      err_msg = f'Expected {dtype=} for {path} but got {x_dtype}.'
      return err_msg
    return None

  err_msgs = jax.tree.leaves(
      jax.tree_util.tree_map_with_path(_assert_dtypes_equal, tree)
  )
  err_msgs = [err_msg for err_msg in err_msgs if err_msg is not None]
  if err_msgs:
    raise ValueError('\n'.join(err_msgs))


def _lower_dtype(
    dtype1: jax.typing.DTypeLike, dtype2: jax.typing.DTypeLike
) -> jax.typing.DTypeLike:
  """Returns lower dtype among two dtypes, if any can be promoted to the other.

  Args:
    dtype1: The first dtype to compare.
    dtype2: The second dtype to compare.

  Returns:
    The lowest of the two dtypes, if any can be promoted to the other.

  Raises:
    ValueError: If none of the dtypes can be promoted to the other.
  """
  if jnp.promote_types(dtype1, dtype2) == dtype1:
    return dtype2
  if jnp.promote_types(dtype1, dtype2) == dtype2:
    return dtype1
  raise ValueError(
      f'Cannot compare dtype of {dtype1=} and {dtype2=}.'
      f' Neither {dtype1} nor {dtype2} can be promoted to the other.'
  )


def _higher_dtype(
    dtype1: jax.typing.DTypeLike, dtype2: jax.typing.DTypeLike
) -> jax.typing.DTypeLike:
  """Returns higher dtype among two dtypes, if any can be promoted to the other.

  Args:
    dtype1: The first dtype to compare.
    dtype2: The second dtype to compare.

  Returns:
    The highest of the two dtypes, if any can be promoted to the other.

  Raises:
    ValueError: If none of the dtypes can be promoted to the other.
  """
  if _lower_dtype(dtype1, dtype2) == dtype1:
    return dtype2
  else:
    return dtype1
