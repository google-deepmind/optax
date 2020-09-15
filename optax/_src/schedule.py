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
"""JAX Schedules.

Schedules may be used to anneal the value of a hyper-parameter over time; for
instance, they may be used to anneal the learning rate used to update an agent's
parameters or the exploration factor used to select actions.
"""

from typing import Dict, Union

from absl import logging
import jax.numpy as jnp


def constant_schedule(value: Union[float, int]):
  """Constructs a constant schedule.

  Args:
    value: value to be held constant throughout.

  Returns:
    schedule: A function that maps step counts to values.
  """
  return lambda count: value


def polynomial_schedule(
    init_value: Union[float, int],
    end_value: Union[float, int],
    power: Union[float, int],
    transition_steps: int,
    transition_begin: int = 0):
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
        'value; this will result in a constant schedule with value '
        '`init_value`.')
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'An exponential schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.')
    transition_begin = 0

  else:
    def schedule(count):
      count = jnp.clip(count - transition_begin, 0, transition_steps)
      frac = 1 - count / transition_steps
      return (init_value - end_value) * (frac**power) + end_value
    return schedule


def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: Dict[int, float] = None):
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
          'The `piecewise_constant_schedule` expects non-negative scale '
          'factors')

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
    staircase: bool = False):
  """Constructs a schedule with either cointinuous or discrete exponential decay.

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

  Returns:
    schedule: A function that maps step counts to values.
  """

  if transition_steps <= 0:
    logging.info(
        'An exponential schedule was set with a non-positive `transition_steps` '
        'value; this will result in a constant schedule with value '
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

  def schedule(count):
    count -= transition_begin
    p = count / transition_steps
    if staircase:
      p = jnp.floor(p)
    return jnp.where(
        count <= 0, init_value, init_value * jnp.power(decay_rate, p))

  return schedule


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0):
  """Returns a function which implements cosine learning rate decay.

  For more details see:
    https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx

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
