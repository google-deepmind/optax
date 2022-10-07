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
"""JAX Schedules.

Schedules may be used to anneal the value of a hyper-parameter over time; for
instance, they may be used to anneal the learning rate used to update an agent's
parameters or the exploration factor used to select actions.
"""

import functools
import inspect
from typing import Callable, Dict, Union, NamedTuple, Optional, Iterable, Sequence

from absl import logging
import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import numerics


def constant_schedule(
    value: Union[float, int]
) -> base.Schedule:
  """Constructs a constant schedule.

  Args:
    value: value to be held constant throughout.

  Returns:
    schedule: A function that maps step counts to values.
  """
  return lambda count: value


def polynomial_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    power: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0
) -> base.Schedule:
  """Constructs a schedule with polynomial transition from init to end value.

  Args:
    init_value: initial value for the scalar to be annealed.
    end_value: end value of the scalar to be annealed.
    power: the power of the polynomial used to transition from init to end.
    transition_steps: number of steps over which annealing takes place,
      the scalar starts changing at `transition_begin` steps and completes
      the transition by `transition_begin + transition_steps` steps.
      If `transition_steps <= 0`, then the entire annealing process is disabled
      and the value is held fixed at `init_value`.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at `init_value`).

  Returns:
    schedule: A function that maps step counts to values.
  """
  if transition_steps <= 0:
    logging.info(
        'A polynomial schedule was set with a non-positive `transition_steps` '
        'value; this results in a constant schedule with value `init_value`.')
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'An exponential schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.')
    transition_begin = 0

  def schedule(count):
    count = jnp.clip(count - transition_begin, 0, transition_steps)
    frac = 1 - count / transition_steps
    return (init_value - end_value) * (frac**power) + end_value
  return schedule


# Alias polynomial schedule to linear schedule for convenience.
def linear_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0
) -> base.Schedule:
  return polynomial_schedule(
      init_value=init_value, end_value=end_value, power=1,
      transition_steps=transition_steps, transition_begin=transition_begin)


def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: Optional[Dict[int, float]] = None
) -> base.Schedule:
  """Returns a function which implements a piecewise constant schedule.

  Args:
    init_value: An initial value `init_v`.
    boundaries_and_scales: A map from boundaries `b_i` to non-negative scaling
      factors `f_i`. For any step count `s`, the schedule returns `init_v`
      scaled by the product of all factors `f_i` such that `b_i` < `s`.

  Returns:
    schedule: A function that maps step counts to values.
  """
  if boundaries_and_scales is not None:
    all_positive = all(scale >= 0. for scale in boundaries_and_scales.values())
    if not all_positive:
      raise ValueError(
          '`piecewise_constant_schedule` expects non-negative scale factors')

  def schedule(count):
    v = init_value
    if boundaries_and_scales is not None:
      for threshold, scale in sorted(boundaries_and_scales.items()):
        indicator = jnp.maximum(0., jnp.sign(threshold - count))
        v = v * indicator + (1 - indicator) * scale * v
    return v

  return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None
) -> base.Schedule:
  """Constructs a schedule with either continuous or discrete exponential decay.

  This function applies an exponential decay function to a provided initial
  value. The function returns the decayed value as follows:

  ```
  decayed_value = init_value * decay_rate ^ (count / transition_steps)
  ```

  If the argument `staircase` is `True`, then `count / transition_steps` is
  an integer division and the decayed value follows a staircase function.

  Args:
    init_value: the initial learning rate.
    transition_steps: must be positive. See the decay computation above.
    decay_rate: must not be zero. The decay rate.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at `init_value`).
    staircase: if `True`, decay the values at discrete intervals.
    end_value: the value at which the exponential decay stops. When
      `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
      an upper bound. Has no effect when `decay_rate` = 0.

  Returns:
    schedule: A function that maps step counts to values.
  """

  if transition_steps <= 0:
    logging.info(
        'An exponential schedule was set with a non-positive `transition_steps`'
        ' value; this will result in a constant schedule with value '
        '`init_value`.')
    return lambda count: init_value

  if decay_rate == 0:
    logging.info(
        'An exponential schedule was set with a zero `decay_rate` value; '
        'this will result in a constant schedule with value `init_value`.')
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'An exponential schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.')
    transition_begin = 0

  if end_value is not None:
    clip_fn = jnp.maximum if decay_rate < 1.0 else jnp.minimum

  def schedule(count):
    decreased_count = count - transition_begin
    p = decreased_count / transition_steps
    if staircase:
      p = jnp.floor(p)
    decayed_value = jnp.where(
        decreased_count <= 0, init_value, init_value * jnp.power(decay_rate, p))
    if end_value is not None:
      decayed_value = clip_fn(decayed_value, end_value)
    return decayed_value

  return schedule


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0
) -> base.Schedule:
  """Returns a function which implements cosine learning rate decay.

  The schedule does not restart when ``decay_steps`` has been reached. Instead,
  the learning rate remains constant afterwards. For a cosine schedule with
  restarts, :func:`optax.join_schedules` can be used to join several cosine
  decay schedules.

  For more details see: https://arxiv.org/abs/1608.03983.

  Args:
    init_value: An initial value `init_v`.
    decay_steps: Positive integer - the number of steps for which to apply
      the decay for.
    alpha: Float. The minimum value of the multiplier used to adjust the
      learning rate.

  Returns:
    schedule: A function that maps step counts to values.
  """
  if not decay_steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive decay_steps!')

  def schedule(count):
    count = jnp.minimum(count, decay_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return init_value * decayed

  return schedule


def _linear_interpolate(start: float, end: float, pct: float):
  return (end-start) * pct + start


def _cosine_interpolate(start: float, end: float, pct: float):
  return end + (start-end) / 2.0 * (jnp.cos(jnp.pi * pct) + 1)


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Optional[Dict[int, float]] = None
) -> base.Schedule:
  """Returns a function which implements a piecewise interpolated schedule.

  Args:
    interpolate_type: 'linear' or 'cosine', specifying the interpolation
      strategy.
    init_value: An initial value `init_v`.
    boundaries_and_scales: A map from boundaries `b_i` to non-negative scaling
      factors `f_i`. At boundary step `b_i`, the schedule returns `init_v`
      scaled by the product of all factors `f_j` such that `b_j` <= `b_i`. The
      values in between each boundary will be interpolated as per `type`.

  Returns:
    schedule: A function that maps step counts to values.
  """
  if interpolate_type == 'linear':
    interpolate_fn = _linear_interpolate
  elif interpolate_type == 'cosine':
    interpolate_fn = _cosine_interpolate
  else:
    raise ValueError('`interpolate_type` must be either \'cos\' or \'linear\'')

  if boundaries_and_scales:
    boundaries, scales = zip(*sorted(boundaries_and_scales.items()))
    if not all(scale >= 0. for scale in scales):
      raise ValueError(
          '`piecewise_interpolate_schedule` expects non-negative scale factors')
  else:
    boundaries, scales = (), ()

  bounds = jnp.stack((0,) + boundaries)
  values = jnp.cumprod(jnp.stack((init_value,) + scales))
  interval_sizes = (bounds[1:] - bounds[:-1])

  def schedule(count):
    indicator = (bounds[:-1] <= count) & (count < bounds[1:])
    pct = (count - bounds[:-1]) / interval_sizes
    interp_vals = interpolate_fn(values[:-1], values[1:], pct)
    return indicator.dot(interp_vals) + (bounds[-1] <= count) * values[-1]

  return schedule


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
) -> base.Schedule:
  """Returns a function which implements the onecycle learning rate schedule.

  This function uses a linear annealing strategy.
  For more details see: https://arxiv.org/abs/1708.07120

  Args:
    transition_steps: Number of steps over which annealing takes place.
    peak_value: Maximum value attained by schedule at pct_start percent
      of the cycle (in number of steps).
    pct_start: The percentage of the cycle (in number of steps) spent
      increasing the learning rate.
    pct_final: The percentage of the cycle (in number of steps) spent
      increasing to peak_value then decreasing back to init_value.
    div_factor: Determines the initial value via init_value =
      peak_value / div_factor
    final_div_factor: Determines the final value via final_value =
      init_value / final_div_factor

  Returns:
    schedule: A function that maps step counts to values.
  """
  if transition_steps <= 0:
    raise ValueError(
        'A linear onecycle schedule was set with a non-positive '
        '`transition_steps`')

  return piecewise_interpolate_schedule(
      'linear',
      peak_value / div_factor,
      {int(pct_start * transition_steps): div_factor,
       int(pct_final * transition_steps): 1. / div_factor,
       transition_steps: 1. / final_div_factor})


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
) -> base.Schedule:
  """Returns a function which implements the onecycle learning rate schedule.

  This function uses a cosine annealing strategy.
  For more details see: https://arxiv.org/abs/1708.07120

  Args:
    transition_steps: Number of steps over which annealing takes place.
    peak_value: Maximum value attained by schedule at pct_start percent
      of the cycle (in number of steps).
    pct_start: The percentage of the cycle (in number of steps) spent
      increasing the learning rate.
    div_factor: Determines the initial value via init_value =
      peak_value / div_factor
    final_div_factor: Determines the final value via final_value =
      init_value / final_div_factor

  Returns:
    schedule: A function that maps step counts to values.
  """
  if transition_steps <= 0:
    raise ValueError(
        'A linear onecycle schedule was set with a non-positive '
        '`transition_steps`')

  return piecewise_interpolate_schedule(
      'cosine',
      peak_value / div_factor,
      {int(pct_start * transition_steps): div_factor,
       int(transition_steps): 1. / (div_factor * final_div_factor)})


def join_schedules(schedules: Sequence[base.Schedule],
                   boundaries: Sequence[int]) -> base.Schedule:
  """Sequentially apply multiple schedules.

  Args:
    schedules: A list of callables (expected to be optax schedules). Each
      schedule will receive a step count indicating the number of steps since
      the previous boundary transition.
    boundaries: A list of integers (of length one less than schedules) that
      indicate when to transition between schedules.
  Returns:
    schedule: A function that maps step counts to values.
  """
  def schedule(step: jnp.DeviceArray) -> jnp.DeviceArray:
    output = schedules[0](step)
    for boundary, schedule in zip(boundaries, schedules[1:]):
      output = jnp.where(step < boundary, output, schedule(step - boundary))
    return output
  return schedule


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> base.Schedule:
  """Linear warmup followed by cosine decay.

  Args:
    init_value: Initial value for the scalar to be annealed.
    peak_value: Peak value for scalar to be annealed at end of warmup.
    warmup_steps: Positive integer, the length of the linear warmup.
    decay_steps: Positive integer, the total length of the schedule. Note that
      this includes the warmup time, so the number of steps during which cosine
      annealing is applied is `decay_steps - warmup_steps`.
    end_value: End value of the scalar to be annealed.
  Returns:
    schedule: A function that maps step counts to values.
  """
  schedules = [
      linear_schedule(
          init_value=init_value,
          end_value=peak_value,
          transition_steps=warmup_steps),
      cosine_decay_schedule(
          init_value=peak_value,
          decay_steps=decay_steps - warmup_steps,
          alpha=end_value/peak_value)]
  return join_schedules(schedules, [warmup_steps])


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None
) -> base.Schedule:
  """Linear warmup followed by exponential decay.

  Args:
    init_value: Initial value for the scalar to be annealed.
    peak_value: Peak value for scalar to be annealed at end of warmup.
    warmup_steps: Positive integer, the length of the linear warmup.
    transition_steps: must be positive. See `exponential_decay` for more
      details.
    decay_rate: must not be zero. The decay rate.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at `peak_value`).
    staircase: if `True`, decay the values at discrete intervals.
    end_value: the value at which the exponential decay stops. When
      `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
      an upper bound. Has no effect when `decay_rate` = 0.
  Returns:
    schedule: A function that maps step counts to values.
  """
  schedules = [
      linear_schedule(
          init_value=init_value,
          end_value=peak_value,
          transition_steps=warmup_steps),
      exponential_decay(
          init_value=peak_value,
          transition_steps=transition_steps,
          decay_rate=decay_rate,
          transition_begin=transition_begin,
          staircase=staircase,
          end_value=end_value)]
  return join_schedules(schedules, [warmup_steps])


def sgdr_schedule(cosine_kwargs: Iterable[Dict[str, chex.Numeric]]
                  ) -> base.Schedule:
  """SGD with warm restarts, from Loschilov & Hutter (arXiv:1608.03983).

  This learning rate schedule applies multiple joined cosine decay cycles.
  For more details see: https://arxiv.org/abs/1608.03983

  Args:
    cosine_kwargs: An Iterable of dicts, where each element specifies the
      arguments to pass to each cosine decay cycle. The `decay_steps` kwarg
      will specify how long each cycle lasts for, and therefore when to
      transition to the next cycle.
  Returns:
    schedule: A function that maps step counts to values.
  """
  boundaries = []
  schedules = []
  step = 0
  for kwargs in cosine_kwargs:
    schedules += [warmup_cosine_decay_schedule(**kwargs)]
    boundaries += [step + kwargs['decay_steps']]
    step += kwargs['decay_steps']
  return join_schedules(schedules, boundaries[:-1])


def _convert_floats(x, dtype):
  """Convert float-like inputs to dtype, rest pass through."""
  if jax.dtypes.scalar_type_of(x) == float:
    return jnp.asarray(x, dtype=dtype)
  return x


class InjectHyperparamsState(NamedTuple):
  """Maintains inner transform state, hyperparameters, and step count."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32
  hyperparams: Dict[str, chex.Numeric]
  inner_state: base.OptState


def inject_hyperparams(
    inner_factory: Callable[..., base.GradientTransformation],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[jnp.dtype] = None,
) -> Callable[..., base.GradientTransformation]:
  """Wrapper that injects hyperparameters into the inner GradientTransformation.

  This wrapper allows you to pass schedules (i.e. a function that returns a
  numeric value given a step count) instead of constants for
  hyperparameters. You may only schedule numeric hyperparameters (i.e. boolean
  flags cannot be scheduled).

  For example, to use ``scale_by_adam`` with a piecewise linear
  schedule for beta_1 and constant for beta_2::

    scheduled_adam = optax.inject_hyperparams(optax.scale_by_adam)(
        b1=optax.piecewise_linear_schedule(...),
        b2=0.99)

  You may manually change numeric hyperparameters that were not scheduled
  through the ``hyperparams`` dict in the ``InjectHyperparamState``::

    state = scheduled_adam.init(params)
    updates, state = scheduled_adam.update(grads, state)
    state.hyperparams['b2'] = 0.95
    updates, state = scheduled_adam.update(updates, state)  # uses b2 = 0.95

  Manually overriding scheduled hyperparameters will have no effect (e.g.
  in the code sample above, you cannot manually adjust ``b1``).

  Args:
    inner_factory: a function that returns the inner
      ``optax.GradientTransformation`` given the hyperparameters.
    static_args: a string or iterable of strings specifying which
      callable parameters are not schedules. inject_hyperparams treats all
      callables as schedules by default, so if a hyperparameter is a
      non-schedule callable, you must specify that using this argument.
    hyperparam_dtype: Optional datatype override. If specified, all float
      hyperparameters will be cast to this type.

  Returns:
    A callable that returns a ``optax.GradientTransformation``. This callable
    accepts the same arguments as ``inner_factory``, except you may provide
    schedules in place of the constant arguments.
  """
  static_args = ({static_args} if isinstance(static_args, str) else
                 set(static_args))
  inner_signature = inspect.signature(inner_factory)

  if not static_args.issubset(inner_signature.parameters):
    raise ValueError(
        '`static_args` must specify a subset of `inner_factory`\'s parameters. '
        f'Given `static_args`: {static_args}. `inner_factory` parameters: '
        f'{set(inner_signature.parameters.keys())}')

  @functools.wraps(inner_factory)
  def wrapped_transform(*args, **kwargs) -> base.GradientTransformation:
    bound_arguments = inner_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    sched_hps, numeric_hps, other_hps = {}, {}, {}
    for name, value in bound_arguments.arguments.items():
      if name in static_args or isinstance(value, bool):
        other_hps[name] = value
      elif callable(value):
        sched_hps[name] = value
      elif isinstance(value, (int, float, chex.Array)):
        numeric_hps[name] = value
      else:
        other_hps[name] = value

    def schedule_fn(count, dtype):
      return {k: _convert_floats(f(count), dtype) for k, f in sched_hps.items()}

    def init_fn(params):
      count = jnp.zeros([], jnp.int32)
      if hyperparam_dtype is None:
        dtype = getattr(next(iter(
            jax.tree_util.tree_leaves(params)), None), 'dtype', None)
      else:
        dtype = hyperparam_dtype
      hparams = {
          k: jnp.asarray(_convert_floats(v, dtype))
          for k, v in numeric_hps.items()}
      hparams.update(schedule_fn(count, dtype))
      return InjectHyperparamsState(  # pylint:disable=too-many-function-args
          count, hparams, inner_factory(**other_hps, **hparams).init(params))

    def update_fn(updates, state, params=None):
      if hyperparam_dtype is None:
        dtype = getattr(next(iter(
            jax.tree_util.tree_leaves(updates)), None), 'dtype', None)
      else:
        dtype = hyperparam_dtype
      hparams = {k: _convert_floats(v, dtype)
                 for k, v in state.hyperparams.items()}
      hparams.update(schedule_fn(state.count, dtype))
      updates, inner_state = inner_factory(**other_hps, **hparams).update(
          updates, state.inner_state, params)
      count_inc = numerics.safe_int32_increment(state.count)

      # pylint:disable=too-many-function-args
      return updates, InjectHyperparamsState(count_inc, hparams, inner_state)
      # pylint:enable=too-many-function-args

    return base.GradientTransformation(init_fn, update_fn)

  return wrapped_transform
