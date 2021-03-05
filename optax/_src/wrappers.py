# Lint as: python3
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
"""Transformation wrappers."""

from typing import Any, Callable, NamedTuple, Tuple, Union

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
import numpy as np

from optax._src import transform

Array = jnp.ndarray


def flatten(
    inner: transform.GradientTransformation
) -> transform.GradientTransformation:
  """Flattens parameters and gradients for init and update of inner transform.

  This can reduce the overhead of performing many calculations on lots of small
  variables, at the cost of slightly increased memory usage.

  Args:
    inner: Inner transformation to flatten inputs for.

  Returns:
    New GradientTransformation.
  """

  def _flatten(params):
    """Flattens and concatenates all tensors in params to a single vector."""
    params, _ = tree_flatten(params)
    return jnp.concatenate([jnp.reshape(param, [-1]) for param in params])

  def _unflatten(updates, flat):
    """Extracts tensors from flat, using the structure and shapes of params."""
    updates_flat, treedef = tree_flatten(updates)
    offsets = []
    for update in updates_flat:
      size = np.prod(update.shape)
      if offsets:
        offsets.append(size + offsets[-1])
      else:
        offsets.append(size)
    del offsets[-1]
    flat_split = jnp.split(flat, offsets)
    reshaped = [
        jnp.reshape(flat_update, update.shape)
        for flat_update, update in zip(flat_split, updates_flat)
    ]
    return tree_unflatten(treedef, reshaped)

  def init_fn(params):
    flat = _flatten(params)
    return inner.init(flat)

  def update_fn(updates, state, params=None):
    if params is not None:
      params = _flatten(params)
    updates_flat, state = inner.update(_flatten(updates), state, params)
    updates = _unflatten(updates, updates_flat)
    return updates, state

  return transform.GradientTransformation(init_fn, update_fn)


class ApplyIfFiniteState(NamedTuple):
  """State of the `GradientTransformation` returned by `apply_if_finite`.

  Fields:
    notfinite_count: Number of consecutive gradient updates containing an Inf or
      a NaN. This number is reset to 0 whenever a gradient update without an Inf
      or a NaN is done.
    last_finite: Whether or not the last gradient update contained an Inf of a
      NaN.
    total_notfinite: Total number of gradient updates containing an Inf or
      a NaN since this optimiser was initialised. This number is never reset.
    inner_state: The state of the inner `GradientTransformation`.
  """
  notfinite_count: jnp.array
  last_finite: jnp.array
  total_notfinite: jnp.array
  inner_state: Any


def apply_if_finite(
    inner: transform.GradientTransformation,
    max_consecutive_errors: int
) -> transform.GradientTransformation:
  """A function that wraps an optimiser to make it robust to a few NaNs or Infs.

  The purpose of this function is to prevent any optimisation to happen if the
  gradients contain NaNs or Infs. That is, when a NaN of Inf is detected in the
  gradients, the wrapped optimiser ignores that gradient update. If the NaNs or
  Infs persist after a given number of updates, the wrapped optimiser gives up
  and accepts the update.

  Args:
    inner: Inner transformation to be wrapped.
    max_consecutive_errors: Maximum number of consecutive gradient updates
      containing NaNs of Infs that the wrapped optimiser will ignore. After
      that many ignored updates, the optimiser will give up and accept.

  Returns:
    New GradientTransformation.
  """

  def init(params):
    return ApplyIfFiniteState(
        notfinite_count=jnp.zeros([], jnp.int64),
        last_finite=jnp.array(True, jnp.bool_),
        total_notfinite=jnp.zeros([], jnp.int64),
        inner_state=inner.init(params))

  def update(updates, state, params=None):
    inner_state = state.inner_state
    flat_updates = tree_flatten(updates)[0]
    isfinite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates]))
    notfinite_count = jnp.where(isfinite, jnp.zeros([], jnp.int64),
                                1 + state.notfinite_count)

    def do_update(_):
      return inner.update(updates, inner_state, params)
    def reject_update(_):
      return (tree_map(jnp.zeros_like, updates), inner_state)

    updates, new_inner_state = lax.cond(
        jnp.logical_or(isfinite, notfinite_count > max_consecutive_errors),
        do_update, reject_update, operand=None)

    return updates, ApplyIfFiniteState(
        notfinite_count=notfinite_count,
        last_finite=isfinite,
        total_notfinite=jnp.logical_not(isfinite) + state.total_notfinite,
        inner_state=new_inner_state)

  return transform.GradientTransformation(init=init, update=update)


def _zeros_tree_like(inp_tree):
  return jax.tree_map(jnp.zeros_like, inp_tree)


class MultiStepsState(NamedTuple):
  """State of the `GradientTransformation` returned by `MultiSteps`.

  Fields:
    mini_step: current mini-step counter. At an update, this either increases by
      1 or is reset to 0.
    gradient_step: gradient step counter. This only increases after enough
      mini-steps have been accumulated.
    inner_opt_state: the state of the wrapped otpimiser.
    acc_grads: accumulated gradients over multiple mini-steps.
  """
  mini_step: Array
  gradient_step: Array
  inner_opt_state: Any
  acc_grads: Any


class MultiSteps:
  """An optimiser wrapper to spread gradient computation over multiple steps.

  This wrapper will allow multiple mini-steps to accumulate their gradients
  together before applying them. It wraps another optimiser, and makes sure that
  this optimiser updates its state only when enough mini-steps have been
  performed. At any other mini-step, the inner optimiser is not used and the
  updates returned by the wrapper are all 0.

  The number of mini-steps per gradient update is controlled by a function, and
  it can vary over training. This offers a mean of varying batch size over
  training.
  """

  def __init__(self,
               opt: transform.GradientTransformation,
               every_k_schedule: Union[int, Callable[[Array], Array]],
               use_grad_mean: bool = True):
    """Initialiser.

    Args:
      opt: the wrapped optimiser.
      every_k_schedule: an int or f a function.
        * As a function, it returns how many mini-steps should be accumulated
          in a single gradient step. Its only argument is the current
          gradient step count. By varying the returned value, users can vary the
          overall training batch size.
        * If an `int`, this is the constant number of mini-steps per gradient
          update.
      use_grad_mean: if `True` (the default), gradients accumulated over
        multiple mini-steps are averaged. Otherwise, they are summed.
    """
    self._opt = opt
    if isinstance(every_k_schedule, int):
      self._every_k_schedule = lambda step: every_k_schedule
    else:
      self._every_k_schedule = every_k_schedule
    self._use_grad_mean = use_grad_mean

  @property
  def inner_opt(self):
    return self._opt

  def init(self, params: Any) -> MultiStepsState:
    init_state = MultiStepsState(mini_step=jnp.zeros([], dtype=jnp.int64),
                                 gradient_step=jnp.zeros([], dtype=jnp.int64),
                                 inner_opt_state=self._opt.init(params),
                                 acc_grads=_zeros_tree_like(params))
    return init_state

  def update(self, grads: Any, state: MultiStepsState, params: Any = None):
    """Accumulates gradients and proposes non-zero updates every `k_steps`."""
    del params
    k_steps = self._every_k_schedule(state.gradient_step)
    acc_grads = jax.tree_util.tree_multimap(lambda a, b: a + b, grads,
                                            state.acc_grads)

    def final_step(args):
      del args
      if self._use_grad_mean:
        grads_for_update = jax.tree_map(lambda x: x / k_steps, acc_grads)
      else:
        grads_for_update = acc_grads
      updates, new_inner_state = self._opt.update(
          grads_for_update, state.inner_opt_state)
      new_state = MultiStepsState(mini_step=jnp.zeros([], dtype=jnp.int64),
                                  gradient_step=state.gradient_step + 1,
                                  inner_opt_state=new_inner_state,
                                  acc_grads=_zeros_tree_like(acc_grads))
      return updates, new_state

    def mid_step(args):
      del args
      updates = _zeros_tree_like(grads)
      new_state = MultiStepsState(mini_step=state.mini_step + 1,
                                  gradient_step=state.gradient_step,
                                  inner_opt_state=state.inner_opt_state,
                                  acc_grads=acc_grads)
      return updates, new_state

    updates, new_state = jax.lax.cond(
        state.mini_step < k_steps - 1, (), mid_step, (), final_step)
    return updates, new_state

  def has_updated(self, state: MultiStepsState) -> Array:
    return jnp.logical_and(state.mini_step == 0, state.gradient_step > 0)

  def gradient_transformation(self) -> transform.GradientTransformation:
    return transform.GradientTransformation(init=self.init, update=self.update)


class LookaheadState(transform.OptState):
  """State of the `GradientTransformation` returned by `lookahead`.

  Attributes:
    fast_state: Optimizer state of the fast optimizer.
    steps_since_sync: Number of fast optimizer steps taken since slow and fast
      parameters were synchronized.
  """
  fast_state: transform.OptState
  steps_since_sync: jnp.ndarray


class LookaheadParams(NamedTuple):
  """Holds a pair of slow and fast parameters for the lookahead optimizer.

  Gradients should always be calculated with the fast parameters. The slow
  parameters should be used for testing and inference as they generalize better.
  See the reference for a detailed discussion.

  References:
    [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

  Attributes:
    fast: Fast parameters.
    slow: Slow parameters.
  """
  fast: transform.Params
  slow: transform.Params

  @classmethod
  def init_synced(cls, params: transform.Params) -> 'LookaheadParams':
    """Initialize a pair of synchronized lookahead parameters."""
    return cls(slow=params, fast=params)


def lookahead(fast_optimizer: transform.GradientTransformation,
              sync_period: int,
              slow_step_size: float,
              reset_state: bool = False) -> transform.GradientTransformation:
  """Lookahead optimizer.

  Performs steps with a fast optimizer and periodically updates a set of slow
  parameters. Optionally resets the fast optimizer state after synchronization
  by calling the init function of the fast optimizer.

  Updates returned by the lookahead optimizer should not be modified before they
  are applied, otherwise fast and slow parameters are not synchronized
  correctly.

  References:
    [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

  Args:
    fast_optimizer: The optimizer to use in the inner loop of lookahead.
    sync_period: Number of fast optimizer steps to take before synchronizing
      parameters. Must be >= 1.
    slow_step_size: Step size of the slow parameter updates.
    reset_state: Whether to reset the optimizer state of the fast opimizer after
      each synchronization.

  Returns:
    A `GradientTransformation` with init and update functions. The updates
    passed to the update function should be calculated using the fast lookahead
    parameters only.
  """
  if sync_period < 1:
    raise ValueError('Synchronization period must be >= 1.')

  def init_fn(params: transform.Params) -> LookaheadState:
    try:
      fast_params = params.fast
    except AttributeError:
      # Allowing init_fn to be called with fast parameters reduces the
      # modifications necessary to adapt code to use lookahead in some cases.
      logging.warning(
          '`params` has no attribute `fast`. Continuing by assuming that '
          'only fast parameters were passed to lookahead init.')
      fast_params = params

    return LookaheadState(
        fast_state=fast_optimizer.init(fast_params),
        steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32))

  def update_fn(
      updates: transform.Updates, state: LookaheadState,
      params: LookaheadParams) -> Tuple[LookaheadParams, LookaheadState]:
    updates, fast_state = fast_optimizer.update(updates, state.fast_state,
                                                params)

    sync_next = (state.steps_since_sync == sync_period - 1)
    updates = _lookahead_update(updates, sync_next, params, slow_step_size)
    if reset_state:
      # Jittable way of resetting the fast optimizer state if parameters will be
      # synchronized after this update step.
      initial_state = fast_optimizer.init(params.fast)
      fast_state = jax.tree_multimap(
          lambda current, init: (1 - sync_next) * current + sync_next * init,
          fast_state, initial_state)

    steps_since_sync = (state.steps_since_sync + 1) % sync_period
    return updates, LookaheadState(fast_state, steps_since_sync)

  return transform.GradientTransformation(init_fn, update_fn)


def _lookahead_update(
    updates: transform.Updates, sync_next: bool, params: LookaheadParams,
    slow_step_size: float) -> transform.GradientTransformation:
  """Returns the updates corresponding to one lookahead step.

  References:
    [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

  Args:
    updates: Updates returned by the fast optimizer.
    sync_next: Wether fast and slow parameters should be synchronized after the
      fast optimizer step.
    params: Current fast and slow parameters as `LookaheadParams` object.
    slow_step_size: Step size of the slow optimizer.

  Returns:
    The updates for the lookahead parameters.
  """
  # In the paper, lookahead is presented as two nested loops. To write lookahead
  # as optax wrapper, these loops have to be broken into successive updates.
  # This leads to two types of update steps:
  #
  # Non-synchronization steps (sync_next == False):
  # The updates returned by the fast optimizer are used for the fast parameters
  # without change and the slow parameter updates are zero (i.e. fast_updates =
  # updates, slow_updates = 0).
  #
  # Synchronisation step (sync_next == True):
  # This consists of two substeps: a last fast optimizer step and the
  # synchronization.
  #   Substep 1 (last fast optimizer step):
  #     last_fast_params = fast_params + updates
  #   Substep 2 (synchronization):
  #     new_slow_params = slow_params + slow_step_size * (
  #                       last_fast_params - slow_params)
  #     new_fast_params = new_slow_params
  #
  #   Merging into a single update step we get the update rules:
  #     slow_updates = slow_step_size * (fast_params + updates - slow_params)
  #     fast_updates = new_slow_params - fast_params = updates - (1 -
  #       slow_step_size) * (fast_params + updates - slow_params)
  #
  # To make the equations jittable, the two types of steps are merged. Defining
  # last_difference = fast_params + updates - slow_params, this yields the
  # following equtions which are implemented below:
  #   slow_updates = slow_step_size * sync_next * last_difference
  #   fast_updates = updates - (
  #                  1 - slow_step_size) * sync_next * last_difference
  last_difference = jax.tree_multimap(lambda f, u, s: f + u - s, params.fast,
                                      updates, params.slow)
  slow_updates = jax.tree_map(lambda diff: slow_step_size * sync_next * diff,
                              last_difference)
  fast_updates = jax.tree_multimap(
      lambda up, diff: up - sync_next * (1 - slow_step_size) * diff, updates,
      last_difference)

  return LookaheadParams(fast=fast_updates, slow=slow_updates)


class MaskedState(NamedTuple):
  """Maintains inner transform state for masked transformations."""
  inner_state: Any


def masked(inner: transform.GradientTransformation,
           mask: Any) -> transform.GradientTransformation:
  """Mask updates so only a subset of them are computed.

  For example, it is common to skip weight decay for BatchNorm scale and all
  bias parameters. In many networks, these are the only parameters with only
  one dimension. So, you may mask these out as follows:

  ```
  mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
  custom_weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)
  ```

  For the `inner` transform, state will only be stored for the parameters that
  have a mask value of `True`.

  Args:
    inner: Inner transformation to mask.
    mask: A PyTree with the same structure as the parameters or is a prefix of
      the parameter PyTree. The leaves should be booleans which are `True` for
      leaves/subtrees you want to apply the transformation to, and `False` for
      those you want to skip.

  Returns:
    New GradientTransformation wrapping `inner`.
  """
  flat_mask, treedef = tree_flatten(mask)

  def init_fn(params):
    flat_params = treedef.flatten_up_to(params)
    masked_params = [p for p, m in zip(flat_params, flat_mask) if m]
    return MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None):
    # Flatten then filter out updates/params not in the mask:
    flat_updates = treedef.flatten_up_to(updates)
    masked_updates = [g for g, m in zip(flat_updates, flat_mask) if m]

    if params:
      flat_params = treedef.flatten_up_to(params)
      masked_params = [p for p, m in zip(flat_params, flat_mask) if m]
    else:
      masked_params = None

    # Compute new updates
    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params)

    # Incorporate new_masked_updates into flat_updates, then unflatten
    new_masked_updates = iter(new_masked_updates)
    for i, m in enumerate(flat_mask):
      if m: flat_updates[i] = next(new_masked_updates)

    new_updates = treedef.unflatten(flat_updates)
    return new_updates, MaskedState(inner_state=new_inner_state)

  return transform.GradientTransformation(init_fn, update_fn)


class MaybeUpdateState(NamedTuple):
  """Maintains inner transform state and adds a step counter."""
  inner_state: Any
  step: Array


def maybe_update(
    inner: transform.GradientTransformation,
    should_update_fn: Callable[[Array], Array]
) -> transform.GradientTransformation:
  """Calls the inner update function only at certain steps.

  Creates a transformation wrapper which counts the number of times the `update`
  function has been called. This counter is passed to the `should_update_fn` to
  decide when to call the inner update function.

  When not calling the inner update function, the `updates` and the inner state
  are left untouched and just passed through. The step counter is increased
  regardless.

  Args:
    inner: the inner transformation.
    should_update_fn: this function takes in a step counter (array of shape []
      and dtype int64), and returns a boolean array of shape [].

  Returns:
    An `optax.GradientTransformation`.
  """

  def init_fn(params):
    return MaybeUpdateState(inner_state=inner.init(params),
                            step=jnp.zeros([], dtype=jnp.int64))

  def update_fn(updates, state, params=None):

    def do_update(_):
      return inner.update(updates, state.inner_state, params)

    def reject_update(_):
      return updates, state.inner_state

    updates, new_inner_state = lax.cond(
        should_update_fn(state.step), do_update, reject_update, operand=None)
    return updates, MaybeUpdateState(new_inner_state, state.step + 1)

  return transform.GradientTransformation(init_fn, update_fn)
