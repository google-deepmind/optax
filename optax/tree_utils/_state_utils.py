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
"""Tools for mapping over optimizer states."""

import typing
from typing import Any, Callable, Hashable, Optional, Protocol, Union, cast

import jax
from optax._src import base

_JaxKeyType = Union[
    int,
    str,
    Hashable,
    jax.tree_util.SequenceKey,
    jax.tree_util.DictKey,
    jax.tree_util.FlattenedIndexKey,
    jax.tree_util.GetAttrKey,
]


@typing.runtime_checkable
class Initable(Protocol):
  """An object with an init function."""

  def init(self, params: base.Params) -> base.OptState:
    """Calling the init for given parameters returns a fresh opt state."""


def tree_map_params(
    initable: Union[
        Callable[[base.Params], base.OptState],
        Initable,
    ],
    f: Callable[..., Any],
    state: base.OptState,
    /,
    *rest: Any,
    transform_non_params: Optional[Callable[..., Any]] = None,
    is_leaf: Optional[Callable[[base.Params], bool]] = None,
) -> base.OptState:
  """Apply a callable over all params in the given optimizer state.

  This function exists to help construct partition specs over optimizer
  states, in the case that a partition spec is already known for the parameters.

  For example, the following will replace all optimizer state parameter trees
  with copies of the given partition spec instead. The argument
  `transform_non_params` can be used to replace any remaining fields as
  required, in this case, we replace those fields by None.

  >>> params, specs = jnp.array(0.), jnp.array(0.)  # Trees with the same shape
  >>> opt = optax.sgd(1e-3)
  >>> state = opt.init(params)
  >>> opt_specs = optax.tree_map_params(
  ...     opt,
  ...     lambda _, spec: spec,
  ...     state,
  ...     specs,
  ...     transform_non_params=lambda _: None,
  ...     )

  Args:
    initable: A callable taking parameters and returning an optimizer state, or
      an object with an `init` attribute having the same function.
    f: A callable that will be applied for all copies of the parameter tree
      within this optimizer state.
    state: The optimizer state to map over.
    *rest: Additional arguments, having the same shape as the parameter tree,
      that will be passed to f.
    transform_non_params: An optional function that will be called on all
      non-parameter fields within the optimizer state.
    is_leaf: Passed through to `jax.tree_map`. This makes it possible to ignore
      parts of the parameter tree e.g. when the gradient transformations modify
      the shape of the original pytree, such as for ``optax.masked``.

  Returns:
    The result of applying the function f on all trees in the optimizer's state
    that have the same shape as the parameter tree, along with the given
    optional extra arguments.
  """

  # Cast for pytype checks (no-op for other usages).
  placeholder = cast(base.chex.ArrayTree, _ParamsPlaceholder())

  if isinstance(initable, Initable):
    initable = cast(Initable, initable)  # for pytype checks
    state_with_placeholders = initable.init(placeholder)
  else:
    state_with_placeholders = initable(placeholder)

  def map_params(maybe_placeholder_value, value):
    if isinstance(maybe_placeholder_value, _ParamsPlaceholder):
      return jax.tree_map(f, value, *rest, is_leaf=is_leaf)
    elif transform_non_params is not None:
      return transform_non_params(value)
    else:
      return value

  return jax.tree_map(
      map_params,
      state_with_placeholders,
      state,
      is_leaf=lambda v: isinstance(v, _ParamsPlaceholder),
  )


def _convert_jax_key_fn(key: _JaxKeyType) -> Union[int, str]:
  """Convert a key returned by `jax.tree_util` to a usual type."""
  if isinstance(key, (str, int)):
    return key  # int | str.
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx  # int.
  if isinstance(key, jax.tree_util.DictKey):
    if isinstance(key.key, (str, int)):
      return key.key
    raise KeyError("Hashable keys not supported")
  if isinstance(key, jax.tree_util.FlattenedIndexKey):
    return key.key  # int.
  if isinstance(key, jax.tree_util.GetAttrKey):
    return key.name  # str.
  raise KeyError(f"Jax tree key '{key}' of type '{type(key)}' not valid.")


def tree_get_all_with_path(
    tree: base.PyTree,
    key: Any,
) -> list[tuple[jax._src.tree_util.KeyPath, Any]]:
  r"""Extract values from leaves of a pytree matching a given key.

  Search in the leaves of a pytree for a specific ``key`` (which can be a key
  from a dictionary or a name from a NamedTuple for example).
  That key or name may appear more than once in the pytree. So this function
  returns a list of all values corresponding to ``key`` with the path to
  that value.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> params = jnp.array([1., 2., 3.])
    >>> base_opt = optax.chain(
    ...   optax.adam(learning_rate=1.),
    ...   optax.adam(learning_rate=1.)
    ... )
    >>> solver = optax.chain(optax.adam(learning_rate=1.), base_opt)
    >>> state = solver.init(params)
    >>> values_found = optax.tree_utils.tree_get_all_with_path(state, 'count')
    >>> print(len(values_found))
    3
    >>> path_to_count, count = values_found[0]
    >>> print(path_to_count, count)
    (SequenceKey(idx=0), SequenceKey(idx=0), GetAttrKey(name='count')) 0

  .. seealso:: :func:`optax.tree_utils.tree_get`

  Args:
    tree: tree to search in.
    key: keyword or name to search in tree for.

  Returns:
    values_with_path
      list of tuples where each tuple is of the form
      (``path_to_value``, ``value``). Here ``value`` is one entry of the state
      that corresponds to the ``key``, and ``path_to_value`` is a path returned
      by :func:`jax.tree_util.tree_flatten_with_path`.

  Raises:
    ValueError: If the input tree is flat, i.e., it is not a tuple/list/dict.
  """
  found_values_with_path = []
  tree_flatten_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
  if not tree_flatten_with_path or not tree_flatten_with_path[0][0]:
    raise ValueError(
        "The input tree cannot be flat, i.e., it must be a tuple/list/dict."
    )
  for path, val in tree_flatten_with_path:
    key_leaf = _convert_jax_key_fn(path[-1])
    if key_leaf == key:
      found_values_with_path.append((path, val))
  return found_values_with_path


def tree_get(tree: base.PyTree, key: Any, default: Optional[Any] = None) -> Any:
  """Extract a value from leaves of a pytree matching a given key.

  Search in the leaves of a pytree for a specific ``key`` (which can be a key
  from a dictionary or a name from a NamedTuple).

  If no leaves in the tree have the required ``key`` returns ``default``.

  Raises a ``KeyError`` if multiple values of ``key`` are found in ``tree``.

  .. seealso:: :func:`optax.tree_utils.tree_get_all_with_path`

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> params = jnp.array([1., 2., 3.])
    >>> solver = optax.inject_hyperparams(optax.adam)(learning_rate=1.)
    >>> state = solver.init(params)
    >>> lr = optax.tree_utils.tree_get(state, 'learning_rate')
    >>> print(lr)
    1.0

  Args:
    tree: tree to search in.
    key: keyword or name to search in tree for.
    default: default value to return if no leaves in the tree matched the given
      ``key``.

  Returns:
    value
      value in the tree matching the given ``key``. If none are
      found return default value. If multiple are found raises an error.

  Raises:
    KeyError: If multiple values of ``key`` are found in ``tree``.
    ValueError: If the input tree is flat, i.e., it is not a tuple/list/dict.
  """
  found_values_with_path = tree_get_all_with_path(tree, key)
  if len(found_values_with_path) > 1:
    raise KeyError(f"Found multiple values for '{key}' in {tree}.")
  elif not found_values_with_path:
    return default
  else:
    return found_values_with_path[0][1]


def tree_set(tree: base.PyTree, **kwargs: Any) -> base.PyTree:
  """Creates a copy of tree with some leaves replaced as specified by kwargs.

  Raises a ``KeyError`` if some keys in ``kwargs`` are not present in the tree.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> params = jnp.array([1., 2., 3.])
    >>> opt = optax.inject_hyperparams(optax.adam)(learning_rate=1.)
    >>> state = opt.init(params)
    >>> new_state = optax.tree_utils.tree_set(state, learning_rate=2.)
    >>> lr = optax.tree_utils.tree_get(new_state, 'learning_rate')
    >>> print(lr)
    2.0

  Args:
    tree: pytree whose values are to be replaced.
    **kwargs: dictionary of keys with values to replace in the tree.

  Returns:
    new_tree
      new pytree with the same structure as tree. For each leaf whose
      key/name matches a key in ``**kwargs``, their values are set by the
      corresponding value in ``**kwargs``.

  Raises:
    KeyError: If no values of some key in ``**kwargs`` are found in ``tree``.
    ValueError: If the input tree is flat, i.e., it is not a tuple/list/dict.
  """
  tree_flatten_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
  if not tree_flatten_with_path or not tree_flatten_with_path[0][0]:
    raise ValueError(
        "The input tree cannot be flat, i.e., it must be a tuple/list/dict."
    )
  key_leaves = [
      _convert_jax_key_fn(path[-1]) for path, _ in tree_flatten_with_path
  ]
  if (left_keys := set(kwargs) - set(key_leaves)):
    left_keys_str = " nor ".join({f"'{key}'" for key in left_keys})
    raise KeyError(f"Found no value for {left_keys_str} in {tree}.")

  def _replace(path, value):
    """Replace a value in tree if key from path matches some key in kwargs."""
    key_leaf = _convert_jax_key_fn(path[-1])
    if key_leaf in kwargs:
      return kwargs[key_leaf]
    else:
      return value

  return jax.tree_util.tree_map_with_path(_replace, tree)


@jax.tree_util.register_pytree_node_class
class _ParamsPlaceholder:

  def tree_flatten(self):
    return ((), None)

  @classmethod
  def tree_unflatten(cls, aux, children):
    del aux, children
    return cls()
