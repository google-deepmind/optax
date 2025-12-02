# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Monitoring and debugging gradient transformations."""

from typing import Any, NamedTuple, Callable

import chex
import jax
from optax._src import base
from optax.transforms import _accumulation
from optax.transforms import _combining


class SnapshotState(NamedTuple):
  measurement: dict[str, Any]


def snapshot(
    measure_name: str, measure: Callable[[base.Updates], chex.ArrayTree]
) -> base.GradientTransformation:
  """Takes a snapshot of updates and stores it in the state.

  Useful to debug intermediate updates values in a chained transformation.

  Args:
    measure_name: Name of the measurement to store. Can be then used to retrieve
      the snapshot using `optax.tree.get(state, measure_name)`.
    measure: User callable taking as inputs updates and returning desired
      measurement. When this transformation is part of a chain, the updates are
      the transformed gradients up to that transform.

  Returns:
    A gradient transformation that captures measurements defined by the user in
    the callable `measure` and stores them in the state with the name
    `measure_name`.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> solver = optax.chain(
    ...     optax.sgd(learning_rate=0.1, momentum=0.9),
    ...     optax.snapshot('norm_before_clip', lambda x: optax.tree.norm(x)),
    ...     optax.clip_by_global_norm(0.05)
    ... )
    >>> params = jnp.array([1., 2., 3.])
    >>> state = solver.init(params)
    >>> for step in range(2):
    ...   grads = jax.grad(f)(params)
    ...   updates, state = solver.update(grads, state)
    ...   params = optax.apply_updates(params, updates)
    ...   norm = optax.tree.get(state, 'norm_before_clip')
    ...   print(f'{step=}, {norm=:.2e}')
    step=0, norm=7.48e-01
    step=1, norm=1.41e+00

  .. versionadded: 0.2.6
  """

  def init(params: base.Params) -> SnapshotState:
    return SnapshotState({measure_name: measure(params)})

  def update(
      updates: base.Updates,
      state: SnapshotState,
      params: base.Params | None = None,
  ) -> tuple[base.Updates, SnapshotState]:
    del params, state
    return updates, SnapshotState({measure_name: measure(updates)})

  return base.GradientTransformation(init, update)


class MonitorState(NamedTuple):
  measurements: dict[str, chex.ArrayTree]
  measure_states: tuple[base.OptState, ...]


def monitor(
    measures: dict[
        str,
        base.GradientTransformationExtraArgs
        | Callable[[base.Updates], chex.ArrayTree],
    ],
):
  """Monitors stateful measurements of updates in a chain.

  Extends func::`optax.snaphot` to use stateful measurements, such as using
  exponential moving average.

  Args:
    measures: A dictionary of measurement names to gradient transformations
      capturing them.

  Returns:
    A gradient transformation that captures measurements defined by the user.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> clip_thresh = 1.0
    >>> solver = optax.chain(
    ...     optax.sgd(learning_rate=0.1, momentum=0.9),
    ...     optax.monitor({
    ...         'norm_before_clip': optax.tree.norm,
    ...         'is_clipped_ema': optax.measure_with_ema(
    ...             lambda x: optax.tree.norm(x) > clip_thresh,
    ...             decay=0.9,
    ...         )
    ...     }),
    ...     optax.clip_by_global_norm(clip_thresh),
    ... )
    >>> params = jnp.array([1., 2., 3.])
    >>> state = solver.init(params)
    >>> for step in range(2):
    ...   grads = jax.grad(f)(params)
    ...   updates, state = solver.update(grads, state)
    ...   params = optax.apply_updates(params, updates)
    ...   norm_before_clip = optax.tree.get(state, 'norm_before_clip')
    ...   is_clipped_ema = optax.tree.get(state, 'is_clipped_ema')
    ...   print(f'{step=}, {norm_before_clip=:.2e}, {is_clipped_ema=:.2e}')
    step=0, norm_before_clip=7.48e-01, is_clipped_ema=0.00e+00
    step=1, norm_before_clip=1.27e+00, is_clipped_ema=5.26e-01

  .. versionadded: 0.2.7
  """

  measures_ = {}
  for measure_name, measure in measures.items():
    if callable(measure):
      measure_ = base.stateless(lambda u, _, m=measure: m(u))
      measures_[measure_name] = base.with_extra_args_support(measure_)
    else:
      measures_[measure_name] = base.with_extra_args_support(measure)
  measures = measures_
  measure_names = tuple(measures.keys())

  def init(params: base.Params) -> MonitorState:
    measurements = {}
    measure_states = []
    for measure_name in measure_names:
      measure_states.append(measures[measure_name].init(params))
    return MonitorState(measurements, tuple(measure_states))

  def update(
      updates: base.Updates,
      state: MonitorState,
      params: base.Params | None = None,
      **extra_args: dict[str, Any],
  ) -> tuple[base.Updates, MonitorState]:
    measurements = {}
    new_measure_states = []
    for i, measure_name in enumerate(measure_names):
      measurement, measure_state = measures[measure_name].update(
          updates,
          state.measure_states[i],
          params,
          **extra_args,
      )
      measurements[measure_name] = measurement
      new_measure_states.append(measure_state)
    return updates, MonitorState(measurements, tuple(new_measure_states))

  return base.GradientTransformationExtraArgs(init, update)


def measure_with_ema(
    measure: Callable[[base.Updates], chex.ArrayTree],
    decay: jax.typing.ArrayLike,  # float
    debias: bool = True,
    accumulator_dtype: Any | None = None
) -> base.GradientTransformationExtraArgs:
  """Take a measurement and record it with exponential moving average.

  Args:
    measure: User callable taking as inputs updates and returning desired
      measurement.
    decay: Decay rate for the exponential moving average.
    debias: Whether to debias the exponential moving average.
    accumulator_dtype: Optional dtype for the exponential moving average
      accumulator.

  Returns:
    A gradient transformation that captures measurements defined by the user,
    and records them with exponential moving average.

  .. seealso::
    :func:`optax.monitor`

  .. versionadded: 0.2.7
  """
  base_ema = _accumulation.ema(decay, debias, accumulator_dtype)
  def init_for_measurement(params):
    # ema needs to be initialized with a variable of the shape it will be
    # accumulated in. In this case, it is the shape of the measurement that can
    # be inferred from applying the measure to params.
    return base_ema.init(measure(params))
  measurement_ema = base_ema._replace(init=init_for_measurement)
  return _combining.chain(
      base.stateless(lambda updates, _: measure(updates)),
      measurement_ema
  )
