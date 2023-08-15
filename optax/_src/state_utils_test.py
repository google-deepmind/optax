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
"""Tests for state_utils."""

import dataclasses
from typing import Optional, TypedDict, cast

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import schedule
from optax._src import state_utils
from optax._src import transform


@dataclasses.dataclass
class FakeShardSpec:
  sharding_axis: Optional[int]


class ScaleByAdamStateDict(TypedDict):
  """An opt state that uses dictionaries instead of classes."""

  count: chex.Array
  params: TypedDict('Params', {'mu': chex.ArrayTree, 'nu': chex.ArrayTree})


def _scale_by_adam_with_dicts():
  """An implementation of adam using dictionary-based opt states."""

  t = transform.scale_by_adam()

  def init(params):
    state = t.init(params)
    state = cast(transform.ScaleByAdamState, state)

    return ScaleByAdamStateDict(
        count=state.count,
        params={'mu': state.mu, 'nu': state.nu},
    )

  def update(updates, state, params=None):
    state = transform.ScaleByAdamState(
        count=state['count'],
        mu=state['params']['mu'],
        nu=state['params']['nu'],
    )

    updates, state = t.update(updates, state, params)
    state = cast(transform.ScaleByAdamState, state)
    return ScaleByAdamStateDict(
        count=state.count,
        params={'mu': state.mu, 'nu': state.nu},
    )

  return base.GradientTransformation(init, update)


class StateUtilsTest(absltest.TestCase):

  def test_dict_based_optimizers(self):
    """Test we can map over params also for optimizer states using dicts."""
    opt = combine.chain(
        _scale_by_adam_with_dicts(),
        transform.additive_weight_decay(1e-3),
    )

    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()
    opt_state = opt.init(params)

    opt_state_sharding_spec = state_utils.tree_map_params(
        opt,
        lambda _, spec: spec,
        opt_state,
        params_sharding_spec,
        transform_non_params=lambda _: FakeShardSpec(None),
    )

    expected = (
        {
            'count': FakeShardSpec(sharding_axis=None),
            'params': {
                'mu': {
                    'my/fake/module': {
                        'b': FakeShardSpec(sharding_axis=1),
                        'w': FakeShardSpec(sharding_axis=0),
                    },
                    'my/other/fake/module': {
                        'b': FakeShardSpec(sharding_axis=3),
                        'w': FakeShardSpec(sharding_axis=2),
                    },
                },
                'nu': {
                    'my/fake/module': {
                        'b': FakeShardSpec(sharding_axis=1),
                        'w': FakeShardSpec(sharding_axis=0),
                    },
                    'my/other/fake/module': {
                        'b': FakeShardSpec(sharding_axis=3),
                        'w': FakeShardSpec(sharding_axis=2),
                    },
                },
            },
        },
        base.EmptyState(),
    )

    self.assertEqual(expected, opt_state_sharding_spec)

  def test_state_chex_dataclass(self):

    @chex.dataclass
    class Foo:
      count: int
      v: chex.ArrayTree

    def init(params):
      return Foo(count=0, v=params)

    params = {
        'w': 0,
    }

    state = init(params)
    state = state_utils.tree_map_params(init, lambda v: v+1, state)
    state = cast(Foo, state)

    self.assertEqual(int(state.count), 0)
    self.assertEqual(state.v, {'w': jnp.array(1)})

  def test_adam(self):
    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()

    opt = alias.adam(1e-4)
    opt_state = opt.init(params)

    opt_state_sharding_spec = state_utils.tree_map_params(
        opt,
        lambda _, spec: spec,
        opt_state,
        params_sharding_spec,
        transform_non_params=lambda _: FakeShardSpec(None),
    )

    expected = (
        transform.ScaleByAdamState(  # pytype:disable=wrong-arg-types
            count=FakeShardSpec(sharding_axis=None),
            mu={
                'my/fake/module': {
                    'w': FakeShardSpec(sharding_axis=0),
                    'b': FakeShardSpec(sharding_axis=1),
                },
                'my/other/fake/module': {
                    'w': FakeShardSpec(sharding_axis=2),
                    'b': FakeShardSpec(sharding_axis=3),
                },
            },
            nu={
                'my/fake/module': {
                    'w': FakeShardSpec(sharding_axis=0),
                    'b': FakeShardSpec(sharding_axis=1),
                },
                'my/other/fake/module': {
                    'w': FakeShardSpec(sharding_axis=2),
                    'b': FakeShardSpec(sharding_axis=3),
                },
            },
        ),
        base.EmptyState(),
    )

    self.assertEqual(expected, opt_state_sharding_spec)

  def test_inject_hparams(self):
    opt = schedule.inject_hyperparams(alias.adamw)(learning_rate=1e-3)

    params = _fake_params()
    state = opt.init(params)
    state = state_utils.tree_map_params(opt, lambda v: v+1, state)
    state = cast(schedule.InjectHyperparamsState, state)

    self.assertEqual(1e-3, state.hyperparams['learning_rate'])
    params_plus_one = jax.tree_map(lambda v: v+1, params)
    mu = getattr(state.inner_state[0], 'mu')
    chex.assert_trees_all_close(mu, params_plus_one)

  def test_map_params_to_none(self):
    opt = alias.adagrad(1e-4)

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)
    state = state_utils.tree_map_params(opt, lambda _: None, state)
    self.assertEqual(
        state,
        (
            transform.ScaleByRssState(sum_of_squares={'a': None}),
            base.EmptyState(),
        ),
    )

  def test_map_non_params_to_none(self):
    """Test for dangerous edge-cases in tree when returning None values."""

    opt = alias.adam(schedule.linear_schedule(1e-2, 1e-4, 10))

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)

    state = state_utils.tree_map_params(
        opt,
        lambda v: 1, state, transform_non_params=lambda _: None
    )

    expected = (
        transform.ScaleByAdamState(  # pytype:disable=wrong-arg-types
            count=None,
            mu={'a': 1},
            nu={'a': 1},
        ),
        transform.ScaleByScheduleState(  # pytype:disable=wrong-arg-types
            count=None),
    )
    self.assertEqual(state, expected)


def _fake_params():
  return {
      'my/fake/module': {
          'w': jnp.zeros((1, 2)),
          'b': jnp.zeros((3, 4)),
      },
      'my/other/fake/module': {
          'w': jnp.zeros((1, 2)),
          'b': jnp.zeros((3, 4)),
      },
  }


def _fake_param_sharding():
  return {
      'my/fake/module': {
          'w': FakeShardSpec(0),
          'b': FakeShardSpec(1),
      },
      'my/other/fake/module': {
          'w': FakeShardSpec(2),
          'b': FakeShardSpec(3),
      },
  }


if __name__ == '__main__':
  absltest.main()
