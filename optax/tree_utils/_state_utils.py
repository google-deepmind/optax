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

import functools
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

_JaxKeyPath = jax._src.tree_util.KeyPath  # pylint: disable=protected-access


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


@jax.tree_util.register_pytree_node_class
class _ParamsPlaceholder:

  def tree_flatten(self):
    return ((), None)

  @classmethod
  def tree_unflatten(cls, aux, children):
    del aux, children
    return cls()


def _convert_jax_key_fn(key: _JaxKeyType) -> Union[int, str]:
  """Convert a key returned by `jax.tree_util` to a usual type."""
  if isinstance(key, (str, int)):
    return key
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


def _node_has_keys(node: Any, keys: tuple[Any, ...]) -> bool:
  """Filter for nodes in a tree whose field/key matches the given key.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path` and in
  :func:`optax.tree_utils.tree_set`.

  Args:
    node: node in a pytree.
    keys: keys to search for in the node.

  Returns:
    whether the node has one of the given keys.
  """
  if (
      isinstance(node, tuple)
      and hasattr(node, "_fields")
      and any(key in node._fields for key in keys)
  ):
    return True
  elif isinstance(node, dict) and any(key in node for key in keys):
    return True
  else:
    return False


def _flatten_to_key(
    path: _JaxKeyPath, node: Any, key: Any
) -> tuple[_JaxKeyPath, Any]:
  """Flatten a node with a field/key matching key up to the value of that key.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path`.

  Args:
    path: path to the node in a pytree.
    node: node in a pytree.
    key: key to reach for in the node.

  Returns:
    (path, new_node)
      if key is a key/field of the node, path = (path_to_node, key_path),
      new_node = node[key], otherwise returns the path and node as they are.
  """
  # Check if node is a NamedTuple
  # A NamedTuple is a tuple with an attribute _fields
  if (
      isinstance(node, tuple)
      and hasattr(node, "_fields")
      and (key in node._fields)
  ):
    return (*path, jax.tree_util.GetAttrKey(key)), getattr(node, key)
  # Check if node is a dict
  elif isinstance(node, dict) and key in node:
    return (*path, jax.tree_util.DictKey(key)), node[key]
  else:
    return path, node


def _get_children_with_path(
    path: _JaxKeyType, node: Any
) -> list[tuple[_JaxKeyPath, Any]]:
  """Get children of a node.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path` and in
  :func:`optax.tree_utils.tree_set`. In particular, it is tailored for
  nodes that are NamedTuple or dict

  Args:
    path: path to the node in a pytree.
    node: node in a pytree.

  Returns:
    list of (path_to_child, child) for child a child in nodes.

  Raises:
    ValueError if the given node is not a NamedTuple or a dict
  """
  # Develop children if node is a NamedTuple
  # A NamedTuple is a tuple with an attribute _fields
  if isinstance(node, tuple) and hasattr(node, "_fields"):
    return [
        ((*path, jax.tree_util.GetAttrKey(field)), getattr(node, field))
        for field in node._fields
    ]
  # Develop children if node is a dict
  elif isinstance(node, dict):
    return [
        ((*path, jax.tree_util.DictKey(key)), value)
        for key, value in node.items()
    ]
  else:
    raise ValueError(
        f"Subtree must be a dict or a NamedTuple. Got {type(node)}"
    )


def _set_children(node: _JaxKeyType, children_with_keys: dict[Any, Any]) -> Any:
  """Set children of a node.

  Private method used in :func:`optax.tree_utils.tree_set`.
  In particular, it is tailored for nodes that are NamedTuple or dict

  Args:
    node: node in a pytree.
    children_with_keys: children of the node with associated keys

  Returns:
    new_node whose fields/keys are replaced by the ones given in
    children_with_keys.

  Raises:
    ValueError if the given node is not a NamedTuple or a dict
  """
  if isinstance(node, tuple) and hasattr(node, "_fields"):
    return node._replace(**children_with_keys)
  elif isinstance(node, dict):
    return children_with_keys
  else:
    raise ValueError(
        f"Subtree must be a dict or a NamedTuple. Got {type(node)}"
    )


def _tree_get_all_with_path(
    tree: base.PyTree, key: str
) -> list[tuple[_JaxKeyPath, Any]]:
  """Get all values of a pytree matching a given key.

  Private function called recursively, see
  :func:`optax.tree_utils.tree_get_all_with_path` for public api.

  Args:
    tree: tree to search in.
    key: keyword or name to search in tree for.

  Returns:
    values_with_path
      list of tuples where each tuple is of the form
      (``path_to_value``, ``value``). Here ``value`` is one entry of the state
      that corresponds to the ``key``, and ``path_to_value`` is a path returned
      by :func:`jax.tree_util.tree_flatten_with_path`.
  """

  # Get subtrees containing a field with the given key
  has_key = functools.partial(_node_has_keys, keys=(key,))
  leaves_or_subtrees_with_path = jax.tree_util.tree_leaves_with_path(
      tree, is_leaf=has_key
  )
  subtrees_with_path = [
      (path, leaf_or_subtree)
      for path, leaf_or_subtree in leaves_or_subtrees_with_path
      if has_key(leaf_or_subtree)
  ]

  # Get (path_to_value, value) for the subtrees found
  found_values_with_path = [
      _flatten_to_key(path, subtree, key)
      for path, subtree in subtrees_with_path
  ]

  # Further search in subtrees for additional values
  for path, subtree in subtrees_with_path:
    children_with_path = _get_children_with_path(path, subtree)
    for path, child in children_with_path:
      new_values_with_path = _tree_get_all_with_path(child, key)
      new_values_with_path = [
          ((*path, *new_path), new_value)
          for new_path, new_value in new_values_with_path
      ]
      found_values_with_path += new_values_with_path
  return found_values_with_path


def tree_get_all_with_path(
    tree: base.PyTree,
    key: Any,
    filtering: Optional[Callable[[_JaxKeyPath, Any], bool]] = None,
) -> list[tuple[_JaxKeyPath, Any]]:
  # pylint: disable=line-too-long
  r"""Extract values of a pytree matching a given key.

  Search in a pytree ``tree`` for a specific ``key`` (which can be a key
  from a dictionary or a field from a NamedTuple).

  That key/field ``key`` may appear more than once in ``tree``. So this function
  returns a list of all values corresponding to ``key`` with the path to
  that value.

  This function can return leaves or subtrees of the pytree whose key/field
  match the given ``key``.

  Examples:
    Basic usage
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> solver = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = solver.init(params)
      >>> values_found = optax.tree_utils.tree_get_all_with_path(
      ...   state, 'learning_rate'
      ... )
      >>> print(values_found)
      [((GetAttrKey(name='hyperparams'), DictKey(key='learning_rate')), Array(1., dtype=float32)), ((GetAttrKey(name='hyperparams_states'), DictKey(key='learning_rate')), WrappedScheduleState(count=Array(0, dtype=int32)))]

    Usage with a filtering operation
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> solver = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = solver.init(params)
      >>> filtering = lambda path, value: isinstance(value, tuple)
      >>> values_found = optax.tree_utils.tree_get_all_with_path(
      ...   state, 'learning_rate', filtering
      ... )
      >>> print(values_found)
      [((GetAttrKey(name='hyperparams_states'), DictKey(key='learning_rate')), WrappedScheduleState(count=Array(0, dtype=int32)))]

  .. seealso:: :func:`optax.tree_utils.tree_get`

  Args:
    tree: tree to search in.
    key: keyword or field to search in tree for.
    filtering: optional callable to further filter values in tree that match the
      key. ``filtering`` takes as arguments both the path to the value and the
      value that match the given key.

  Returns:
    values_with_path
      list of tuples where each tuple is of the form
      (``path_to_value``, ``value``). Here ``value`` is one entry of the tree
      that corresponds to the ``key``, and ``path_to_value`` is a path returned
      by :func:`jax.tree_util.tree_flatten_with_path`.
  """
  # pylint: enable=line-too-long
  found_values_with_path = _tree_get_all_with_path(tree, key)
  if filtering:
    found_values_with_path = [
        (path, value)
        for path, value in found_values_with_path
        if filtering(path, value)
    ]
  return found_values_with_path


def tree_get(
    tree: base.PyTree,
    key: Any,
    default: Optional[Any] = None,
    filtering: Optional[Callable[[_JaxKeyPath, Any], bool]] = None,
) -> Any:
  """Extract a value from a pytree matching a given key.

  Search in the ``tree`` for a specific ``key`` (which can be a key
  from a dictionary or a field from a NamedTuple).

  If the ``tree`` does not containt ``key`` returns ``default``.

  Raises a ``KeyError`` if multiple values of ``key`` are found in ``tree``.

  .. seealso:: :func:`optax.tree_utils.tree_get_all_with_path`

  Examples:
    Basic usage
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.adam(learning_rate=1.)
      >>> state = opt.init(params)
      >>> count = optax.tree_utils.tree_get(state, 'count')
      >>> print(count)
      0

    Usage with a filtering operation
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = opt.init(params)
      >>> filtering = lambda path, value: isinstance(value, jnp.ndarray)
      >>> lr = optax.tree_utils.tree_get(
      ...   state, 'learning_rate', filtering=filtering
      ... )
      >>> print(lr)
      1.0

  Args:
    tree: tree to search in.
    key: keyword or field to search in ``tree`` for.
    default: default value to return if ``key`` is not found in ``tree``.
    filtering: optional callable to further filter values in ``tree`` that match
      the ``key``. ``filtering`` takes as arguments both the path to the value
      and the value that match the given ``key``.

  Returns:
    value
      value in ``tree`` matching the given ``key``. If none are
      found return ``default`` value. If multiple are found raises an error.

  Raises:
    KeyError: If multiple values of ``key`` are found in ``tree``.
  """
  found_values_with_path = tree_get_all_with_path(
      tree, key, filtering=filtering
  )
  if len(found_values_with_path) > 1:
    raise KeyError(f"Found multiple values for '{key}' in {tree}.")
  elif not found_values_with_path:
    return default
  else:
    return found_values_with_path[0][1]


def tree_set(
    tree: base.PyTree,
    filtering: Optional[Callable[[_JaxKeyPath, Any], bool]] = None,
    /,
    **kwargs: Any,
) -> base.PyTree:
  # pylint: disable=line-too-long
  r"""Creates a copy of tree with some values replaced as specified by kwargs.

  Raises a ``KeyError`` if some keys in ``kwargs`` are not present in the tree.

  .. note:: The recommended usage to inject hyperparameters schedules is through
    :func:`optax.inject_hyperparams`. This function is a helper for other
    purposes.

  Examples:
    Basic usage
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.adam(learning_rate=1.)
      >>> state = opt.init(params)
      >>> print(state)
      (ScaleByAdamState(count=Array(0, dtype=int32), mu=Array([0., 0., 0.], dtype=float32), nu=Array([0., 0., 0.], dtype=float32)), EmptyState())
      >>> new_state = optax.tree_utils.tree_set(state, count=2.)
      >>> print(new_state)
      (ScaleByAdamState(count=2.0, mu=Array([0., 0., 0.], dtype=float32), nu=Array([0., 0., 0.], dtype=float32)), EmptyState())

    Usage with a filtering operation
      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.inject_hyperparams(optax.sgd)(
      ...     learning_rate=lambda count: 1/(count+1)
      ...  )
      >>> state = opt.init(params)
      >>> print(state)
      InjectStatefulHyperparamsState(count=Array(0, dtype=int32), hyperparams={'learning_rate': Array(1., dtype=float32)}, hyperparams_states={'learning_rate': WrappedScheduleState(count=Array(0, dtype=int32))}, inner_state=(EmptyState(), EmptyState()))
      >>> filtering = lambda path, value: isinstance(value, jnp.ndarray)
      >>> new_state = optax.tree_utils.tree_set(
      ...   state, filtering, learning_rate=jnp.asarray(0.1)
      ... )
      >>> print(new_state)
      InjectStatefulHyperparamsState(count=Array(0, dtype=int32), hyperparams={'learning_rate': Array(0.1, dtype=float32, weak_type=True)}, hyperparams_states={'learning_rate': WrappedScheduleState(count=Array(0, dtype=int32))}, inner_state=(EmptyState(), EmptyState()))

  Args:
    tree: pytree whose values are to be replaced.
    filtering: optional callable to further filter values in ``tree`` that match
      the keys to replace. ``filtering`` takes as arugments both the path to the
      value and the value that match one of the given keys.
    **kwargs: dictionary of keys with values to replace in ``tree``.

  Returns:
    new_tree
      new pytree with the same structure as ``tree``. For each element in
      ``tree`` whose key/filed matches a key in ``**kwargs``, their values are
      set by the corresponding value in ``**kwargs``.

  Raises:
    KeyError: If no values of some key in ``**kwargs`` are found in ``tree``
      or none of the values satisfy the filtering operation.
  """
  # pylint: enable=line-too-long

  # First check if the keys are present in the tree
  for key in kwargs:
    found_values_with_path = tree_get_all_with_path(tree, key, filtering)
    if not found_values_with_path:
      if filtering:
        raise KeyError(
            f"Found no values matching '{key}' given the filtering operation in"
            f" {tree}"
        )
      else:
        raise KeyError(f"Found no values matching '{key}' in {tree}")

  has_any_key = functools.partial(_node_has_keys, keys=tuple(kwargs.keys()))

  def _replace(path: _JaxKeyPath, node: Any) -> Any:
    """Replace a node with a new node whose values are updated."""
    if has_any_key(node):
      # The node contains one of the keys we want to replace
      children_with_path = _get_children_with_path(path, node)
      new_children_with_keys = {}
      for child_path, child in children_with_path:
        # Scan each child of that node
        key = _convert_jax_key_fn(child_path[-1])
        if key in kwargs and (
            filtering is None or filtering(child_path, child)
        ):
          # If the child matches a given key given the filtering operation
          # replaces with the new value
          new_children_with_keys.update({key: kwargs[key]})
        else:
          if (
              isinstance(child, tuple)
              or isinstance(child, dict)
              or isinstance(child, list)
          ):
            # If the child is itself a pytree, further search in the child to
            # replace the given value
            new_children_with_keys.update({key: _replace(child_path, child)})
          else:
            # If the child is just a leaf that does not contain the key or
            # satisfies the filtering operation, just return the child.
            new_children_with_keys.update({key: child})
      return _set_children(node, new_children_with_keys)
    else:
      return node

  return jax.tree_util.tree_map_with_path(_replace, tree, is_leaf=has_any_key)
