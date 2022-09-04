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
"""Support for extra kwargs in a gradient transformation's `init` and `update`.

Some users have the need to condition the behavior of a gradient
transformations on dynamical quantities such as the loss. With this experimental
feature we support passing additional kwargs to `init` and `update`.

We introduce `GradientTransformationWithExtraArgs` as an experimental feature.
You can use the new `named_chain` to combine both old-style and new-style
transformations. We will then monitor users to understand how they use it and
gather feedback from optax users before merging this into the stable API.
"""

from typing import Any, Mapping, Optional, Tuple, Union, NamedTuple

from optax._src import base
import typing_extensions


class InitFnWithExtraArgs(typing_extensions.Protocol):
  """Like `TransformInitFn` but with optional `extra_args`."""

  def __call__(
      self,
      params: base.Params,
      *,
      extra_args: Optional[Mapping[str, Any]] = None,
  ) -> base.OptState:
    """The `init` function."""


class UpdateFnWithExtraArgs(typing_extensions.Protocol):
  """Like `TransformUpdateFn` but with optional `extra_args`."""

  def __call__(
      self,
      updates: base.Updates,
      state: base.OptState,
      params: Optional[base.Params] = None,
      *,
      extra_args: Optional[Mapping[str, Any]] = None,
  ) -> Tuple[base.Updates, base.OptState]:
    """The `update` function."""


class GradientTransformationWithExtraArgs(NamedTuple):
  """A pair of pure functions implementing a gradient transformation.

  GradientTransformationWithExtraArgs is just like GradientTransformation but
  both the `init` and `update` functions accept an additional `extra_args` dict.
  This can be used to provide additional dynamic information that is not
  computed by the GradientTransformation itself (e.g. loss) but that may be
  needed by specific algorithms.
  """
  init: InitFnWithExtraArgs
  update: UpdateFnWithExtraArgs


AnyGradientTransformation = Union[
    base.GradientTransformation, GradientTransformationWithExtraArgs]
NamedTransform = Tuple[str, AnyGradientTransformation]


def named_chain(
    *transforms: NamedTransform) -> GradientTransformationWithExtraArgs:
  """Chains optax gradient transformations.

  The `transforms` are `(name, transformation)` pairs, constituted of a string
  `name` and an associated gradient transformation `transformation`. The
  gradient transformation must be an instance of either
  `GradientTransformation` or `GradientTransformationWithExtraArgs`.

  Each `name` is used as key for the state of the corresponding transformation
  within the `named_chain` state. Thus the state of the gradient transformation
  with a given `name` can be retrieved as `opt_state[name]`.

  The `named_chain` accepts an `extra_args` meta-dictionary whose fields are
  the transformations' names and its values are the corresponding extra_args:

  Example:
    tx = named_chain(('one', tx1), ('two', tx2))

    extra_args={
        'one': {'loss': 0.1},
        'two': {'loss': 0.3, 'temperature': 0.01}}
    tx.init(params, extra_args=extra_args}
    tx.update(grads, state, params, extra_args=extra_args)

    # tx1 receives {'loss': 0.1} as extra_args
    # tx2 receives {'loss': 0.3, 'temperature': 0.01} as extra_args

  If one of the transformations does not need extra_args the corresponding
  name can just be skipped in the `named_chain` extra_args:

  Example:
    tx = named_chain(('one', tx1), ('two', tx2))

    extra_args={'one': {'loss': 0.1}}
    tx.init(params, extra_args=extra_args}
    tx.update(grads, state, params, extra_args=extra_args)

    # tx1 receives {'loss': 0.1} as extra_args.
    # tx2 is called without passing the extra_args.

  Args:
    *transforms: an arbitrary number of `(name, tx)` pairs, constituted of a
      string `name` and an associated gradient transformation `tx`. The latter
      is a `GradientTransformation` or `GradientTransformationWithExtraArgs`.

  Returns:
    A single (init_fn, update_fn) tuple.
  """

  names = [name for name, _ in transforms]
  if len(names) != len(set(names)):
    raise ValueError(
        f'Named transformations must have unique names, but got {names}')

  def init_fn(params, *, extra_args=None):
    states = {}
    for (name, tx) in transforms:
      _assert_is_gradient_transformation(tx)
      if (extra_args is not None and
          isinstance(tx, GradientTransformationWithExtraArgs)):
        states[name] = tx.init(
            params, extra_args=extra_args.get(name))
      else:
        states[name] = tx.init(params)
    return states

  def update_fn(updates, state, params=None, *, extra_args=None):
    new_state = {}
    for (name, tx) in transforms:
      _assert_is_gradient_transformation(tx)
      if (extra_args is not None and
          isinstance(tx, GradientTransformationWithExtraArgs)):
        updates, new_state[name] = tx.update(
            updates, state[name], params, extra_args=extra_args.get(name))
      else:
        updates, new_state[name] = tx.update(updates, state[name], params)
    return updates, new_state

  return GradientTransformationWithExtraArgs(init_fn, update_fn)


def _assert_is_gradient_transformation(tx):
  valid_types = (
      base.GradientTransformation,
      GradientTransformationWithExtraArgs)
  if not isinstance(tx, valid_types):
    raise ValueError(
        'The transformation `tx` must be a valid gradient transformation, '
        'that is an instance of either `GradientTransformation` or '
        'an instance of `GradientTransformationWithExtraArgs`')
