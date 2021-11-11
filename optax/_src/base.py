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
"""Base interfaces and datatypes."""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import typing_extensions


NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.')

PyTree = Any
Shape = Sequence[int]

OptState = chex.ArrayTree  # States are arbitrary nests of `jnp.ndarrays`.
Params = chex.ArrayTree  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.
TransformInitFn = Callable[[Params], OptState]
Schedule = Callable[[chex.Numeric], chex.Numeric]


class TransformUpdateFn(typing_extensions.Protocol):
  """A callable type for computing a new parameters and opt_state pair.

  The params argument is optional, but must be provided when using a
  transformation that requires access to it.
  """

  def __call__(self,
               updates: Updates,
               state: OptState,
               params: Optional[Params] = None) -> Tuple[Updates, OptState]:
    ...


class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: TransformInitFn
  update: TransformUpdateFn


class EmptyState(NamedTuple):
  """An empty state for the simplest stateless transformations."""


def identity() -> GradientTransformation:
  """Stateless identity transformation that leaves input gradients untouched.

  This function passes through the *gradient updates* unchanged.

  Note, this should not to be confused with `set_to_zero`, which maps the input
  updates to zero - which is the transform required for the *model parameters*
  to be left unchanged when the updates are applied to them.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return EmptyState()

  def update_fn(updates, state, params=None):
    del params
    return updates, state

  return GradientTransformation(init_fn, update_fn)


def set_to_zero() -> GradientTransformation:
  """Stateless transformation that maps input gradients to zero.

  The resulting update function, when called, will return a tree of zeros
  matching the shape of the input gradients. This means that when the updates
  returned from this transformation are applied to the model parameters, the
  model parameters will remain unchanged.

  This can be used in combination with `multi_transform` to keep some parts of
  the tree of model parameters fixed while applying gradient updates to other
  parts of the tree.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return EmptyState()

  def update_fn(updates, state, params=None):
    del params  # Unused by the zero transform.
    return jax.tree_map(jnp.zeros_like, updates), state

  return GradientTransformation(init_fn, update_fn)
