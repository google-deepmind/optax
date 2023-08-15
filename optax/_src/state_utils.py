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
from typing import Any, Callable, Optional, Protocol, Union, cast

import jax
from optax._src import base


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

  >>> params, specs = ...  # Trees with the same shape
  >>> state = opt.init(params)
  >>>
  >>> opt_specs = optax.tree_map_params(
  >>>     opt,
  >>>     lambda _, spec: spec,
  >>>     state,
  >>>     specs,
  >>>     transform_non_params=lambda _: None,
  >>> )

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
