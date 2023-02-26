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

from typing import Any, Mapping, Optional, Protocol, Tuple, Union, NamedTuple

from optax._src import base


class InitFnWithExtraArgs(Protocol):
  """Like `TransformInitFn` but with optional `extra_args`."""

  def __call__(
      self,
      params: base.Params,
      *,
      extra_args: Optional[Mapping[str, Any]] = None,
  ) -> base.OptState:
    """The `init` function."""


class UpdateFnWithExtraArgs(Protocol):
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


class ReduceLROnPlateauState(NamedTuple):
    """State for the ReduceLROnPlateau callback."""
    reduce_factor: float
    patience: int
    min_improvement: float
    best_loss: float
    plateau_count: int
    lr: float
    cooldown_counter: int
    cooldown:int


def reduce_on_plateau(
    reduce_factor: float,
    patience: int,
    min_improvement:float,
    cooldown:int
) -> GradientTransformationWithExtraArgs:
    """        Args:
            reduce_factor: Factor by which the learning rate will be reduced. 
                new_lr = lr * factor.
            patience: Number of epochs with no improvement after which learning 
                rate will be reduced.
            min_improvement: Threshold for measuring the new optimum, to only focus on 
                significant changes.
            cooldown: Number of epochs to wait before resuming normal operation 
                after lr has been reduced.
            """


    def init_fn(params):
        del params
        return ReduceLROnPlateauState(patience=patience,
                                      reduce_factor=reduce_factor,
                                      min_improvement=min_improvement,
                                      cooldown=cooldown,
                                      cooldown_counter=0,
                                      plateau_count=0,
                                      best_loss=float("inf"),
                                      lr=1,
                                      )

    def update_fn(
        updates,
        state,
        params=None,
        extra_args={},
    ):
        del params
        current_loss = extra_args.get("loss")

        # Check if the current loss is the best so far

        best_loss = state.best_loss
        # Update plateau count and check if plateaued
        has_improved = jnp.where(
            (current_loss / best_loss - 1) < -state.min_improvement, 1, 0
        )
        new_best_loss = jnp.where(has_improved, current_loss, best_loss)
        
        curr_plateau_count = jnp.where(has_improved, 0, state.plateau_count + 1)
        
        
        # We're in cooldown, so reduce the counter and ignore any bad epochs
        def in_cooldown():
            new_plateau_count = 0
            new_lr = state.lr
            new_cooldown_counter = state.cooldown_counter - 1
            return new_plateau_count, new_lr, new_cooldown_counter

        # We're not in cooldown, so update the plateau count and lr as usual
        def not_in_cooldown():
            new_plateau_count = jnp.where(
                curr_plateau_count == state.patience, 0, curr_plateau_count
            )
            new_lr = jnp.where(
                curr_plateau_count == state.patience,
                state.lr * state.reduce_factor,
                state.lr,
            )
            new_cooldown_counter = jnp.where(
                curr_plateau_count == state.patience, state.cooldown, 0
            )
            return new_plateau_count, new_lr, new_cooldown_counter
        
        new_plateau_count, new_lr, new_cooldown_counter = jax.lax.cond(state.cooldown_counter > 0, in_cooldown, not_in_cooldown)

        updates = jax.tree_util.tree_map(lambda g: new_lr * g, updates)

        new_state = ReduceLROnPlateauState(
            patience=state.patience,
            reduce_factor=state.reduce_factor,
            min_improvement=state.min_improvement,
            plateau_count=new_plateau_count,
            best_loss=new_best_loss,
            lr=new_lr,
            cooldown_counter=new_cooldown_counter,
            cooldown=state.cooldown,
        )
        return updates, new_state

    return GradientTransformationWithExtraArgs(init_fn, update_fn)
