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

import functools
from typing import Any, Callable, NamedTuple, Optional, Protocol, Tuple, Union

import chex
import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
import numpy as np
from optax._src import base
from optax._src import numerics
from optax._src import state_utils


Array = jnp.ndarray


def flatten(
    inner: base.GradientTransformation
) -> base.GradientTransformationExtraArgs:
  """Flattens parameters and gradients for init and update of inner transform.

  This can reduce the overhead of performing many calculations on lots of small
  variables, at the cost of slightly increased memory usage.

  Args:
    inner: Inner transformation to flatten inputs for.

  Returns:
    New ``GradientTransformationExtraArgs``
  """

  inner = base.with_extra_args_support(inner)

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

  def update_fn(updates, state, params=None, **extra_args):
    if params is not None:
      params = _flatten(params)
    updates_flat, state = inner.update(
        _flatten(updates), state, params, **extra_args
    )
    updates = _unflatten(updates, updates_flat)
    return updates, state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class ApplyIfFiniteState(NamedTuple):
  """State of the `GradientTransformation` returned by `apply_if_finite`.

  Fields:
    notfinite_count: Number of consecutive gradient updates containing an Inf or
      a NaN. This number is reset to 0 whenever a gradient update without an Inf
      or a NaN is done.
    last_finite: Whether or not the last gradient update contained an Inf of a
      NaN.
    total_notfinite: Total number of gradient updates containing an Inf or
      a NaN since this optimizer was initialised. This number is never reset.
    inner_state: The state of the inner `GradientTransformation`.
  """
  notfinite_count: jnp.array
  last_finite: jnp.array
  total_notfinite: jnp.array
  inner_state: Any


def apply_if_finite(
    inner: base.GradientTransformation,
    max_consecutive_errors: int
) -> base.GradientTransformation:
  """A function that wraps an optimizer to make it robust to a few NaNs or Infs.

  The purpose of this function is to prevent any optimization to happen if the
  gradients contain NaNs or Infs. That is, when a NaN of Inf is detected in the
  gradients, the wrapped optimizer ignores that gradient update. If the NaNs or
  Infs persist after a given number of updates, the wrapped optimizer gives up
  and accepts the update.

  Args:
    inner: Inner transformation to be wrapped.
    max_consecutive_errors: Maximum number of consecutive gradient updates
      containing NaNs of Infs that the wrapped optimizer will ignore. After
      that many ignored updates, the optimizer will give up and accept.

  Returns:
    New ``GradientTransformationExtraArgs``.
  """

  inner = base.with_extra_args_support(inner)

  def init(params):
    return ApplyIfFiniteState(
        notfinite_count=jnp.zeros([], jnp.int32),
        last_finite=jnp.array(True, jnp.bool_),
        total_notfinite=jnp.zeros([], jnp.int32),
        inner_state=inner.init(params))

  def update(updates, state, params=None, **extra_args):
    inner_state = state.inner_state
    flat_updates = tree_flatten(updates)[0]
    isfinite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates]))
    notfinite_count = jnp.where(
        isfinite, jnp.zeros([], jnp.int32),
        numerics.safe_int32_increment(state.notfinite_count))

    def do_update(_):
      return inner.update(updates, inner_state, params, **extra_args)
    def reject_update(_):
      return (tree_map(jnp.zeros_like, updates), inner_state)

    updates, new_inner_state = lax.cond(
        jnp.logical_or(isfinite, notfinite_count > max_consecutive_errors),
        do_update, reject_update, operand=None)

    return updates, ApplyIfFiniteState(
        notfinite_count=notfinite_count,
        last_finite=isfinite,
        total_notfinite=jnp.where(
            isfinite, state.total_notfinite,
            numerics.safe_int32_increment(state.total_notfinite)),
        inner_state=new_inner_state)

  return base.GradientTransformationExtraArgs(init=init, update=update)


def _zeros_tree_like(inp_tree):
  return jax.tree_util.tree_map(jnp.zeros_like, inp_tree)


class MultiStepsState(NamedTuple):
  """State of the `GradientTransformation` returned by `MultiSteps`.

  Fields:
    mini_step: current mini-step counter. At an update, this either increases by
      1 or is reset to 0.
    gradient_step: gradient step counter. This only increases after enough
      mini-steps have been accumulated.
    inner_opt_state: the state of the wrapped otpimiser.
    acc_grads: accumulated gradients over multiple mini-steps.
    skip_state: an arbitrarily nested tree of arrays. This is only
      relevant when passing a `should_skip_update_fn` to `MultiSteps`. This
      structure will then contain values for debugging and or monitoring. The
      actual structure will vary depending on the choice of
      `ShouldSkipUpdateFunction`.
  """
  mini_step: Array
  gradient_step: Array
  inner_opt_state: Any
  acc_grads: Any
  skip_state: chex.ArrayTree = ()


class ShouldSkipUpdateFunction(Protocol):

  def __call__(self, updates: base.Updates, gradient_step: Array,
               params: Optional[base.Params]) -> Tuple[Array, chex.ArrayTree]:
    """Returns true to indicate that updates should be skipped in a multi-step.

    Args:
      updates: The updates that the gradient transformation has proposed
        to apply
      gradient_step: The current gradient step (see
        `MultiStepsState.gradient_step`). This can be used for example to reject
        large gradients with an annealed maximum allowed gradient norm.
      params: If known, the current parameter tree of the function being
        transformed.
    Returns:
      A tuple:
      * First element is an array with a single bool indicating whether or not
        the updates should be applied.
      * Second element is an arbitrarily nested structure of arrays that will be
        stored in `MultiStepsState.skip_state`. The structure will vary from
        function to function. Debugging info, or values to monitor, can be put
        in this structure.
    """


def skip_not_finite(
    updates: base.Updates, gradient_step: Array,
    params: Optional[base.Params]) -> Tuple[Array, chex.ArrayTree]:
  """Returns True iff any of the `updates` contains an inf or a NaN.

  Args:
    updates: see `ShouldSkipUpdateFunction`.
    gradient_step: see `ShouldSkipUpdateFunction`.
    params: see `ShouldSkipUpdateFunction`.

  Returns:
    A tuple:
    * First element is a scalar array of type bool.
    * Second element is a dictionary with keys:
      - `should_skip`: True iff `updates` contains an inf or a NaN.
      - `num_not_finite`: total number of inf and NaN found in `updates`.
  """
  del gradient_step, params
  all_is_finite = [jnp.sum(jnp.logical_not(jnp.isfinite(p)))
                   for p in jax.tree_util.tree_leaves(updates)]
  num_not_finite = jnp.sum(jnp.array(all_is_finite))
  should_skip = num_not_finite > 0
  return should_skip, dict(should_skip=should_skip,
                           num_not_finite=num_not_finite)


def skip_large_updates(updates: base.Updates,
                       gradient_step: Array,
                       params: Optional[base.Params],
                       max_squared_norm: float) -> Tuple[Array, chex.ArrayTree]:
  """Returns True if the global norm square of `updates` is small enough.

  Args:
    updates: see `ShouldSkipUpdateFunction`.
    gradient_step: see `ShouldSkipUpdateFunction`.
    params: see `ShouldSkipUpdateFunction`.
    max_squared_norm: only updates with a norm square strictly less than this
      value will be accepted.

  Returns:
    A tuple:
    * First element is a scalar array of type bool.
    * Second element is a dictionary with keys:
      - `should_skip`: True iff square norm of `updates` is larger or equal than
        `max_squared_norm`.
      - `norm_squared`: overall norm square of the `updates`.
  """
  del gradient_step, params
  norm_sq = jnp.sum(
      jnp.array([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(updates)]))
  # This will also return True if `norm_sq` is NaN.
  should_skip = jnp.logical_not(norm_sq < max_squared_norm)
  return should_skip, dict(should_skip=should_skip, norm_squared=norm_sq)


class MultiSteps:
  """An optimizer wrapper to accumulate gradients over multiple steps.

  This wrapper collects together the updates passed to its `update` function
  over consecutive steps until a given number of scheduled steps is reached.
  In each of these intermediate steps, the returned value from the optimizer is
  a tree of zeros of the same shape of the updates passed as input.

  Once the scheduled number of intermediate 'mini-steps' has been reached, the
  gradients accumulated to the current time will be passed to the wrapped
  optimizer's update function, (with the inner optimizer's state being updated
  appropriately) and then returned to the caller. The wrapper's accumulated
  gradients are then set back to zero and the process starts again.

  The number of mini-steps per gradient update is controlled by a function, and
  it can vary over training. This offers a means of varying batch size over
  training.
  """

  def __init__(
      self,
      opt: base.GradientTransformation,
      every_k_schedule: Union[int, Callable[[Array], Array]],
      use_grad_mean: bool = True,
      should_skip_update_fn: Optional[ShouldSkipUpdateFunction] = None):
    """Initialiser.

    Args:
      opt: the wrapped optimizer.
      every_k_schedule: an int or f a function.
        * As a function, it returns how many mini-steps should be accumulated
          in a single gradient step. Its only argument is the current
          gradient step count. By varying the returned value, users can vary the
          overall training batch size.
        * If an `int`, this is the constant number of mini-steps per gradient
          update.
      use_grad_mean: if `True` (the default), gradients accumulated over
        multiple mini-steps are averaged. Otherwise, they are summed.
      should_skip_update_fn: if provided, this function is used to decide when
        to accept or reject the updates from a mini-step. When a mini-step is
        rejected, the inner state of `MultiSteps` is not updated. In other
        words, it is as if this mini-step never happened. For example:
        * to ignore updates containing inf or NaN, do
          `should_skip_update_fn=skip_not_finite`;
        * to ignore updates with a norm square larger then 42, do
          `should_skip_update_fn=functools.partial(skip_large_updates,
                                                   max_norm_sq=42.)`.
        Note that the optimizer's state `MultiStepsState` contains a field
        `skip_state` in which debugging and monitoring information returned
        by `should_skip_update_fn` is written.
    """
    self._opt = base.with_extra_args_support(opt)

    if isinstance(every_k_schedule, int):
      self._every_k_schedule = lambda step: every_k_schedule
    else:
      self._every_k_schedule = every_k_schedule
    self._use_grad_mean = use_grad_mean

    if self._use_grad_mean:
      # Use Welford algorithm for numerically stable aggregation of mean.
      self._acc_update = (
          lambda grad, acc, *, n_acc: acc + (grad - acc) / (n_acc + 1))
    else:
      self._acc_update = lambda grad, acc, *, n_acc: grad + acc

    if should_skip_update_fn is None:

      def should_skip_update_fn(*unused_args, **unused_kwargs):
        return jnp.array(False, dtype=jnp.bool_), ()

    self._should_skip_update_fn = should_skip_update_fn

  @property
  def inner_opt(self):
    return self._opt

  def init(self, params: Any) -> MultiStepsState:
    """Builds and returns initial `MultiStepsState`."""
    updates = _zeros_tree_like(params)
    gradient_step = jnp.zeros([], dtype=jnp.int32)
    _, skip_state = self._should_skip_update_fn(updates, gradient_step, params)
    init_state = MultiStepsState(
        mini_step=jnp.zeros([], dtype=jnp.int32),
        gradient_step=gradient_step,
        inner_opt_state=self._opt.init(params),
        acc_grads=updates,
        skip_state=skip_state)
    return init_state

  def update(self,
             updates: base.Updates,
             state: MultiStepsState,
             params: Optional[base.Params] = None,
             **extra_args: Any,
             ) -> Tuple[base.Updates, MultiStepsState]:
    """Accumulates gradients and proposes non-zero updates every `k_steps`."""
    k_steps = self._every_k_schedule(state.gradient_step)
    acc_grads = jax.tree_util.tree_map(
        functools.partial(self._acc_update, n_acc=state.mini_step),
        updates, state.acc_grads)

    should_skip_update, skip_state = self._should_skip_update_fn(
        updates, state.gradient_step, params)

    def final_step(args):
      del args
      final_updates, new_inner_state = self._opt.update(
          acc_grads, state.inner_opt_state, params=params, **extra_args)
      new_state = MultiStepsState(
          mini_step=jnp.zeros([], dtype=jnp.int32),
          gradient_step=numerics.safe_int32_increment(state.gradient_step),
          inner_opt_state=new_inner_state,
          acc_grads=_zeros_tree_like(acc_grads),
          skip_state=skip_state)
      return final_updates, new_state

    def mid_step(args):
      del args
      updates_shape_dtype, _ = jax.eval_shape(
          self._opt.update, acc_grads, state.inner_opt_state, params=params)
      mid_updates = jax.tree_util.tree_map(
          lambda sd: jnp.zeros(sd.shape, sd.dtype), updates_shape_dtype)
      new_state = MultiStepsState(
          mini_step=numerics.safe_int32_increment(state.mini_step),
          gradient_step=state.gradient_step,
          inner_opt_state=state.inner_opt_state,
          acc_grads=acc_grads,
          skip_state=skip_state)
      return mid_updates, new_state

    new_updates, new_state = jax.lax.cond(
        state.mini_step < k_steps - 1, (), mid_step, (), final_step)

    if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
      raise ValueError(
          'The `should_skip_update_fn` function should return a boolean scalar '
          f'array, but it returned an array of dtype {should_skip_update.dtype}'
          f' and shape {should_skip_update.shape}')

    multi_state_when_skip = MultiStepsState(
        mini_step=state.mini_step,
        gradient_step=state.gradient_step,
        inner_opt_state=state.inner_opt_state,
        acc_grads=state.acc_grads,
        skip_state=skip_state)
    zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
    new_updates, new_state = jax.lax.cond(
        should_skip_update,
        (), lambda args: (zero_updates, multi_state_when_skip),
        (), lambda args: (new_updates, new_state))

    return new_updates, new_state

  def has_updated(self, state: Union[MultiStepsState, chex.ArrayTree]) -> Array:
    # Use `getattr` to bypass pytype checks.
    return jnp.logical_and(
        getattr(state, 'mini_step') == 0, getattr(state, 'gradient_step') > 0
    )

  def gradient_transformation(self) -> base.GradientTransformation:
    return base.GradientTransformation(init=self.init, update=self.update)


class MaskedState(NamedTuple):
  """Maintains inner transform state for masked transformations."""
  inner_state: Any


class MaskedNode(NamedTuple):
  """A node used to mask out unspecified parts of a tree.

  This node is ignored when mapping functions across the tree e.g. using
  `jax.tree_util.tree_map` since it is a container without children. It can
  therefore be used to mask out parts of a tree.
  """


def masked(
    inner: base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]]
) -> base.GradientTransformationExtraArgs:
  """Mask updates so only some are transformed, the rest are passed through.

  For example, it is common to skip weight decay for BatchNorm scale and all
  bias parameters. In many networks, these are the only parameters with only
  one dimension. So, you may create a mask function to mask these out as
  follows::

    mask_fn = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)

  You may alternatively create the mask pytree upfront::

    mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)

  For the ``inner`` transform, state will only be stored for the parameters that
  have a mask value of ``True``.

  Note that, when using ``tree_map_params``, it may be required to pass the
  argument `is_leaf=lambda v: isinstance(v, optax.MaskedNode)`, if the tree
  map needs to take additional arguments with the same shape as the original
  input tree.

  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip. The mask must
      be static for the gradient transformation to be jit-compilable.

  Returns:
    New ``GradientTransformationExtraArgs`` wrapping ``inner``.
  """
  inner = base.with_extra_args_support(inner)

  def mask_pytree(pytree, mask_tree):
    return tree_map(lambda m, p: p if m else MaskedNode(), mask_tree, pytree)

  def init_fn(params):
    # This is a workaround to make tree_map_params work with masking.
    # The API of `masked` takes a mask on construction, instead of at init.
    # This means that this gradient transformation can only work for parameter
    # trees that match the shape of the mask. Technically this breaks the API
    # of optax, and this causes tree_map_params to break. This is because
    # tree_map_params calls init with a placeholder in order to detect copies
    # of the parameter tree. As a (slightly ugly) workaround, we detect when
    # the init is being called by tree_map_params, and pass the placeholder
    # down without masking. This is safe, since tree_map_params does not impose
    # any particular constraints on the shape of the parameter tree, as long
    # as tree_map_params is being called on a tree with the correct structure.
    # See wrappers_test for proof that this works!
    if isinstance(params, state_utils._ParamsPlaceholder):  # pylint:disable=protected-access
      return MaskedState(inner_state=inner.init(params))

    mask_tree = mask(params) if callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None, **extra_args):
    mask_tree = mask(updates) if callable(mask) else mask
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)

    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params, **extra_args)

    new_updates = tree_map(
        lambda m, new_u, old_u: new_u if m else old_u,
        mask_tree, new_masked_updates, updates)
    return new_updates, MaskedState(inner_state=new_inner_state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class MaybeUpdateState(NamedTuple):
  """Maintains inner transform state and adds a step counter."""
  inner_state: Any
  step: Array


def maybe_update(
    inner: base.GradientTransformation,
    should_update_fn: Callable[[Array], Array]
) -> base.GradientTransformationExtraArgs:
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
      and dtype int32), and returns a boolean array of shape [].

  Returns:
    A new ``GradientTransformationExtraArgs``.
  """
  inner = base.with_extra_args_support(inner)

  def init_fn(params):
    return MaybeUpdateState(
        inner_state=inner.init(params), step=jnp.zeros([], dtype=jnp.int32))

  def update_fn(updates, state, params=None, **extra_args):

    def do_update(_):
      return inner.update(updates, state.inner_state, params, **extra_args)

    def reject_update(_):
      return updates, state.inner_state

    updates, new_inner_state = lax.cond(
        should_update_fn(state.step), do_update, reject_update, operand=None)
    return updates, MaybeUpdateState(new_inner_state,
                                     numerics.safe_int32_increment(state.step))

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
