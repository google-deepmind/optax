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
"""Tests for `stateful.py`."""

from typing import NamedTuple

from absl.testing import absltest

import chex
import jax.numpy as jnp
import numpy as np

from optax._src import base
from optax._src import transform
from optax.schedules import _schedule
from optax.schedules import _stateful


class ExampleState(NamedTuple):
  total: chex.Numeric


class ExampleStatefulSchedule(base.StatefulSchedule):

  def init(self) -> ExampleState:
    return ExampleState(total=jnp.zeros([], dtype=jnp.int32))

  def update(self, state: ExampleState, **extra_args) -> ExampleState:
    total = state.total + extra_args['addendum']
    return ExampleState(total=total)

  def __call__(self, state: ExampleState, **extra_args) -> chex.Numeric:
    return state.total


class StatefulTest(chex.TestCase):

  def test_wrap_stateless_schedule(self):
    my_schedule = _schedule.linear_schedule(1., 1., 10)
    my_wrapped_schedule = _stateful.WrappedSchedule(my_schedule)

    count = jnp.zeros([], dtype=jnp.int32)
    state = my_wrapped_schedule.init()
    np.testing.assert_allclose(count, state, atol=0.0)

    for _ in range(8):
      np.testing.assert_allclose(
          my_schedule(count), my_wrapped_schedule(state), atol=0.0)
      count = count + 1
      extra_args = dict(loss=jnp.ones([], dtype=jnp.float32))
      state = my_wrapped_schedule.update(state, **extra_args)
      np.testing.assert_allclose(count, state, atol=0.0)

  @chex.all_variants
  def test_inject_stateful_hyperparams(self):
    grads = (
        jnp.ones((3,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),)
    params = grads

    my_stateful_schedule = ExampleStatefulSchedule()
    tx = _stateful.inject_stateful_hyperparams(
        transform.scale)(step_size=my_stateful_schedule)
    state = self.variant(tx.init)(params)

    extra_args = dict(addendum=0.3 * jnp.ones((), dtype=jnp.float32))
    _, state = self.variant(tx.update)(
        grads, state, params=params, **extra_args)
    _, state = self.variant(tx.update)(
        grads, state, params=params, **extra_args)

    lr = state.hyperparams['step_size']
    total = state.hyperparams_states['step_size']

    np.testing.assert_allclose(lr, extra_args['addendum'], atol=0.0)
    np.testing.assert_allclose(total, 2 * extra_args['addendum'], atol=0.0)


if __name__ == '__main__':
  absltest.main()
