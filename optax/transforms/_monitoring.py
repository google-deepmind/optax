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

import jax
from optax._src import base


class SnapshotState(NamedTuple):
  measurement: dict[str, Any]


def snapshot(
    measure_name: str, measure: Callable[[base.Updates,], jax.Array]
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
    ...   print(f'{step=}, {norm=}')
    step=0, norm=Array(0.7483, dtype=float32)
    step=1, norm=Array(1.4118, dtype=float32)

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
