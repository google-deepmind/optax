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

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union
import chex

# pylint:disable=no-value-for-parameter


NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.')

OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.


TransformInitFn = Callable[
    [Params],
    Union[OptState, Sequence[OptState]]]
TransformUpdateFn = Callable[
    [Updates, OptState, Optional[Params]],
    Tuple[Updates, OptState]]
Schedule = Callable[
    [chex.Numeric],
    chex.Numeric]


class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: TransformInitFn
  update: TransformUpdateFn


class EmptyState(OptState):
  """An empty state for the simplest stateless transformations."""


def identity() -> GradientTransformation:
  """Stateless identity transformation that leaves input gradients untouched.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return EmptyState()

  def update_fn(updates, state, params=None):
    del params
    return updates, state

  return GradientTransformation(init_fn, update_fn)

