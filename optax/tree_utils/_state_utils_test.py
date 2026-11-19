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
"""Tests for methods in `optax.tree_utils._state_utils.py`."""

import dataclasses
from typing import Optional, TypedDict, cast
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.sharding import AxisType
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import factorized
from optax._src import test_utils
from optax._src import transform
from optax._src import utils
from optax._src import wrappers
from optax.schedules import _inject
from optax.schedules import _schedule
from optax.tree_utils import _random
from optax.tree_utils import _state_utils


@dataclasses.dataclass
class FakeShardSpec:
  sharding_axis: Optional[int]


class ScaleByAdamStateDict(TypedDict):
  """An opt state that uses dictionaries instead of classes."""

  count: jax.typing.ArrayLike
  params: TypedDict('Params', {'mu': base.ArrayTree, 'nu': base.ArrayTree})


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

    _, state = t.update(updates, state, params)
    state = cast(transform.ScaleByAdamState, state)
    return ScaleByAdamStateDict(
        count=state.count,
        params={'mu': state.mu, 'nu': state.nu},
    )

  return base.GradientTransformation(init, update)


def _make_auto_mesh():
  return jax.sharding.AbstractMesh(
      (2, 4), ('x', 'y'), axis_types=(AxisType.Auto, AxisType.Auto)
  )


def _make_single_device_auto_mesh():
  return jax.make_mesh((1,), ('x',), axis_types=(AxisType.Auto,))


def _sharded_sds(shape, mesh, spec, memory_kind=None, **kwargs):
  """A float32 ShapeDtypeStruct sharded as ``NamedSharding(mesh, spec)``."""
  return jax.ShapeDtypeStruct(
      shape,
      jnp.float32,
      sharding=NamedSharding(mesh, spec, memory_kind=memory_kind),
      **kwargs,
  )


class StateUtilsTest(absltest.TestCase):

  def test_dict_based_optimizers(self):
    """Test we can map over params also for optimizer states using dicts."""
    opt = combine.chain(
        _scale_by_adam_with_dicts(),
        transform.scale(1e-3),  # any empty state transform
    )

    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()
    opt_state = opt.init(params)

    opt_state_sharding_spec = _state_utils.tree_map_params(
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

  def test_adam(self):
    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()

    opt = alias.adam(1e-4)
    opt_state = opt.init(params)

    opt_state_sharding_spec = _state_utils.tree_map_params(
        opt,
        lambda _, spec: spec,
        opt_state,
        params_sharding_spec,
        transform_non_params=lambda _: FakeShardSpec(None),
    )

    expected = (
        transform.ScaleByAdamState(
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
    opt = _inject.inject_hyperparams(alias.adamw)(learning_rate=1e-3)

    params = _fake_params()
    state = opt.init(params)
    state = _state_utils.tree_map_params(opt, lambda v: v + 1, state)
    state = cast(_inject.InjectHyperparamsState, state)

    self.assertEqual(1e-3, state.hyperparams['learning_rate'])
    params_plus_one = jax.tree.map(lambda v: v + 1, params)
    mu = getattr(state.inner_state[0], 'mu')
    test_utils.assert_trees_all_close(mu, params_plus_one)

  def test_map_params_to_none(self):
    opt = alias.adagrad(1e-4)

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)
    state = _state_utils.tree_map_params(opt, lambda _: None, state)
    self.assertEqual(
        state,
        (
            transform.ScaleByRssState(sum_of_squares={'a': None}),
            base.EmptyState(),
        ),
    )

  def test_map_non_params_to_none(self):
    """Test for dangerous edge-cases in tree when returning None values."""

    opt = alias.adam(_schedule.linear_schedule(1e-2, 1e-4, 10))

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)

    state = _state_utils.tree_map_params(
        opt,
        lambda v: 1,
        state,
        transform_non_params=lambda _: None,
    )

    expected = (
        transform.ScaleByAdamState(
            count=None,
            mu={'a': 1},
            nu={'a': 1},
        ),
        transform.ScaleByScheduleState(count=None),
    )
    self.assertEqual(state, expected)

  def test_tree_get_all_with_path(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with flat tree'):
      tree = ()
      found_values = _state_utils.tree_get_all_with_path(tree, 'foo')
      self.assertEmpty(found_values)
      tree = jnp.array([1.0, 2.0, 3.0])
      found_values = _state_utils.tree_get_all_with_path(tree, 'foo')
      self.assertEmpty(found_values)

    with self.subTest('Test with single value in state'):
      key = 'count'
      opt = transform.scale_by_adam()
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      expected_result = [(
          (_state_utils.NamedTupleKey('ScaleByAdamState', 'count'),),
          jnp.array(0.0),
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with no value in state'):
      key = 'apple'
      opt = alias.adam(learning_rate=1.0)
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      self.assertEmpty(found_values)

    with self.subTest('Test with multiple values in state'):
      key = 'learning_rate'
      opt = combine.chain(
          _inject.inject_hyperparams(alias.sgd)(learning_rate=1.0),
          combine.chain(
              alias.adam(learning_rate=1.0),
              _inject.inject_hyperparams(alias.adam)(learning_rate=1e-4),
          ),
      )
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      expected_result = [
          (
              (
                  jtu.SequenceKey(idx=0),
                  _state_utils.NamedTupleKey(
                      'InjectStatefulHyperparamsState', 'hyperparams'
                  ),
                  jtu.DictKey(key='learning_rate'),
              ),
              jnp.array(1.0),
          ),
          (
              (
                  jtu.SequenceKey(idx=1),
                  jtu.SequenceKey(idx=1),
                  _state_utils.NamedTupleKey(
                      'InjectStatefulHyperparamsState', 'hyperparams'
                  ),
                  jtu.DictKey(key='learning_rate'),
              ),
              jnp.array(1e-4),
          ),
      ]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with optional filtering'):
      state = {'hparams': {'learning_rate': 1.0}, 'learning_rate': 'foo'}

      # Without filtering two values are found
      found_values = _state_utils.tree_get_all_with_path(state, 'learning_rate')
      self.assertLen(found_values, 2)

      # With filtering only the float entry is returned
      found_values = _state_utils.tree_get_all_with_path(
          state,
          'learning_rate',
          filtering=lambda _, value: isinstance(value, float),
      )
      self.assertLen(found_values, 1)
      expected_result = [(
          (jtu.DictKey(key='hparams'), jtu.DictKey(key='learning_rate')),
          1.0,
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test to get a subtree (here hyperparams_states)'):
      opt = _inject.inject_hyperparams(alias.sgd)(learning_rate=lambda x: x)
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(
          state,
          'learning_rate',
          filtering=lambda _, value: isinstance(value, tuple),
      )
      expected_result = [(
          (
              _state_utils.NamedTupleKey(
                  'InjectStatefulHyperparamsState', 'hyperparams_states'
              ),
              jtu.DictKey(key='learning_rate'),
          ),
          _inject.WrappedScheduleState(
              count=jnp.array(0, dtype=jnp.dtype('int32'))
          ),
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with nested tree containing a key'):
      tree = {'a': {'a': 1.0}}
      found_values = _state_utils.tree_get_all_with_path(tree, 'a')
      expected_result = [
          ((jtu.DictKey(key='a'),), {'a': 1.0}),
          ((jtu.DictKey(key='a'), jtu.DictKey(key='a')), 1.0),
      ]
      self.assertEqual(found_values, expected_result)

  def test_tree_get(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with unique value matching the key'):
      solver = _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0)
      state = solver.init(params)
      lr = _state_utils.tree_get(state, 'learning_rate')
      self.assertEqual(lr, 42.0)

    with self.subTest('Test with no value matching the key'):
      solver = _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0)
      state = solver.init(params)
      ema = _state_utils.tree_get(state, 'ema')
      self.assertIsNone(ema)
      ema = _state_utils.tree_get(state, 'ema', default=7.0)
      self.assertEqual(ema, 7.0)

    with self.subTest('Test with multiple values matching the key'):
      solver = combine.chain(
          _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0),
          _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0),
      )
      state = solver.init(params)
      self.assertRaises(KeyError, _state_utils.tree_get, state, 'learning_rate')

    with self.subTest('Test jitted tree_get'):
      opt = _inject.inject_hyperparams(alias.sgd)(
          learning_rate=lambda x: 1 / (x + 1)
      )
      state = opt.init(params)

      @jax.jit
      def get_learning_rate(state):
        return _state_utils.tree_get(
            state,
            'learning_rate',
            filtering=lambda _, value: isinstance(value, jnp.ndarray)
        )

      for i in range(4):
        # we simply update state, we don't care about updates.
        _, state = opt.update(params, state)
        lr = get_learning_rate(state)
        self.assertEqual(lr, 1 / (i + 1))

    with self.subTest('Test with optional filtering'):
      state = {'hparams': {'learning_rate': 1.0}, 'learning_rate': 'foo'}

      # Without filtering raises an error
      self.assertRaises(KeyError, _state_utils.tree_get, state, 'learning_rate')

      # With filtering, fetches the float entry
      lr = _state_utils.tree_get(
          state,
          'learning_rate',
          filtering=lambda _, value: isinstance(value, float),
      )
      self.assertEqual(lr, 1.0)

    with self.subTest('Test filtering for specific state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)

      def filtering(path, _):
        return (
            isinstance(path[-1], _state_utils.NamedTupleKey)
            and path[-1].tuple_name == 'ScaleByAdamState'
        )

      count = _state_utils.tree_get(state, 'count', filtering=filtering)
      self.assertEqual(count, jnp.asarray(0, dtype=jnp.dtype('int32')))

    with self.subTest('Test extracting a state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)
      noise_state = _state_utils.tree_get(state, 'AddNoiseState')
      expected_result = transform.AddNoiseState(
          count=jnp.asarray(0),
          rng_key=jnp.array([0, 0], dtype=jnp.dtype('uint32')),
      )
      test_utils.assert_trees_all_equal(
          _random.tree_unwrap_random_key_data(noise_state),
          _random.tree_unwrap_random_key_data(expected_result)
      )

  def test_tree_set(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with flat tree'):
      tree = ()
      self.assertRaises(KeyError, _state_utils.tree_set, tree, foo=1.0)
      tree = jnp.array([1.0, 2.0, 3.0])
      self.assertRaises(KeyError, _state_utils.tree_set, tree, foo=1.0)

    with self.subTest('Test modifying an injected hyperparam'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, learning_rate=2.0, b1=3.0)
      lr = _state_utils.tree_get(new_state, 'learning_rate')
      self.assertEqual(lr, 2.0)

    with self.subTest('Test modifying an attribute of the state'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, learning_rate=2.0, b1=3.0)
      b1 = _state_utils.tree_get(new_state, 'b1')
      self.assertEqual(b1, 3.0)

    with self.subTest('Test modifying a value not present in the state'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      self.assertRaises(KeyError, _state_utils.tree_set, state, ema=2.0)

    with self.subTest('Test jitted tree_set'):

      @jax.jit
      def set_learning_rate(state, lr):
        return _state_utils.tree_set(state, learning_rate=lr)

      modified_state = state
      lr = 1.0
      for i in range(4):
        modified_state = set_learning_rate(modified_state, lr / (i + 1))
        # we simply update state, we don't care about updates.
        _, modified_state = opt.update(params, modified_state)
        modified_lr = _state_utils.tree_get(modified_state, 'learning_rate')
        self.assertEqual(modified_lr, lr / (i + 1))

    with self.subTest('Test modifying several values at once'):
      opt = combine.chain(
          alias.adam(learning_rate=1.0), alias.adam(learning_rate=1.0)
      )
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, count=2.0)
      found_values = _state_utils.tree_get_all_with_path(new_state, 'count')
      self.assertLen(found_values, 2)
      for _, value in found_values:
        self.assertEqual(value, 2.0)

    with self.subTest('Test with optional filtering'):
      state = {'hparams': {'learning_rate': 1.0}, 'learning_rate': 'foo'}
      new_state = _state_utils.tree_set(
          state,
          lambda _, value: isinstance(value, float),
          learning_rate=0.5,
      )
      found_values = _state_utils.tree_get_all_with_path(
          new_state, 'learning_rate'
      )
      expected_result = [
          ((jtu.DictKey(key='learning_rate'),), 'foo'),
          ((jtu.DictKey(key='hparams'), jtu.DictKey(key='learning_rate')), 0.5),
      ]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with nested trees and filtering'):
      tree = {'a': {'a': 1.0}, 'b': {'a': 1}}
      new_tree = _state_utils.tree_set(
          tree,
          lambda _, value: isinstance(value, float),
          a=2.0,
      )
      expected_result = {'a': {'a': 2.0}, 'b': {'a': 1}}
      self.assertEqual(new_tree, expected_result)

    with self.subTest('Test setting a subtree'):
      tree = {'a': {'a': 1.0}, 'b': {'a': 1}}
      new_tree = _state_utils.tree_set(
          tree,
          lambda _, value: isinstance(value, dict),
          a={'c': 0.0},
      )
      expected_result = {'a': {'c': 0.0}, 'b': {'a': 1}}
      self.assertEqual(new_tree, expected_result)

    with self.subTest('Test setting a specific state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)

      def filtering(path, _):
        return (
            isinstance(path[-1], _state_utils.NamedTupleKey)
            and path[-1].tuple_name == 'ScaleByAdamState'
        )

      new_state = _state_utils.tree_set(state, filtering, count=jnp.array(42))
      expected_result = (
          transform.AddNoiseState(
              count=jnp.array(0),
              rng_key=jnp.array([0, 0], dtype=jnp.dtype('uint32')),
          ),
          transform.ScaleByAdamState(
              count=jnp.array(42),
              mu=jnp.array([0.0, 0.0, 0.0]),
              nu=jnp.array([0.0, 0.0, 0.0]),
          ),
      )
      test_utils.assert_trees_all_equal(
          _random.tree_unwrap_random_key_data(new_state),
          _random.tree_unwrap_random_key_data(expected_result)
      )
    with self.subTest('Test setting a state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)
      new_noise_state = transform.AddNoiseState(
          count=jnp.array(42),
          rng_key=jnp.array([4, 8], dtype=jnp.dtype('uint32')),
      )
      new_state = _state_utils.tree_set(state, AddNoiseState=new_noise_state)
      expected_result = (
          transform.AddNoiseState(
              count=jnp.array(42),
              rng_key=jnp.array([4, 8], dtype=jnp.dtype('uint32')),
          ),
          transform.ScaleByAdamState(
              count=jnp.array(0),
              mu=jnp.array([0.0, 0.0, 0.0]),
              nu=jnp.array([0.0, 0.0, 0.0]),
          ),
      )
      test_utils.assert_trees_all_equal(new_state, expected_result)

  def _skip_if_sharding_unsupported(self):
    if utils.parse_version(jax.__version__) < utils.parse_version('0.7.2'):
      self.skipTest('Requires sharding-in-types support (JAX >= 0.7.2).')

  def test_abstract_init_accepts_transform_and_init_callable(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 8), mesh, P('x', 'y')),
    }

    tx = transform.scale_by_adam()
    from_tx = _state_utils.abstract_init(tx, abstract_params)
    from_callable = _state_utils.abstract_init(tx.init, abstract_params)

    self.assertEqual(from_tx, from_callable)
    self.assertEqual(from_tx.mu['w'].sharding, NamedSharding(mesh, P('x', 'y')))
    self.assertEqual(from_tx.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_mesh_types_and_axis_types_preserved(self):
    self._skip_if_sharding_unsupported()
    concrete_mesh = _make_single_device_auto_mesh()

    def abstract_mesh(*axis_types):
      return jax.sharding.AbstractMesh(
          (64, 16), ('x', 'y'), axis_types=axis_types
      )

    # (name, mesh, w_spec, b_spec)
    cases = [
        ('auto_abstract_mesh', abstract_mesh(AxisType.Auto, AxisType.Auto)),
        (
            'explicit_abstract_mesh',
            abstract_mesh(AxisType.Explicit, AxisType.Explicit),
        ),
        (
            'hybrid_auto_explicit_abstract_mesh',
            abstract_mesh(AxisType.Auto, AxisType.Explicit),
        ),
        ('concrete_mesh', concrete_mesh),
    ]

    for name, mesh in cases:
      if mesh is concrete_mesh:
        w_spec, b_spec = P(), P()
      else:
        w_spec, b_spec = P('x', 'y'), P('y')

      with self.subTest(name):
        abstract_params = {
            'w': _sharded_sds((128, 64), mesh, w_spec),
            'b': _sharded_sds((64,), mesh, b_spec),
        }

        opt_state = _state_utils.abstract_init(
            alias.adamw(1e-3), abstract_params
        )
        adam_state = opt_state[0]

        for leaf in jax.tree.leaves(opt_state):
          self.assertIsInstance(leaf, jax.ShapeDtypeStruct)
          self.assertEqual(leaf.sharding.mesh, mesh)

        self.assertEqual(
            adam_state.mu['w'].sharding, NamedSharding(mesh, w_spec)
        )
        self.assertEqual(
            adam_state.mu['b'].sharding, NamedSharding(mesh, b_spec)
        )
        self.assertEqual(adam_state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_same_shape_mixed_dtype_weak_type_and_scalars(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 8), mesh, P('x', 'y'), weak_type=True),
    }
    opt = combine.chain(
        transform.trace(decay=0.9),
        transform.scale_by_adam(mu_dtype=jnp.bfloat16),
        transform.add_decayed_weights(1e-2),
        transform.scale_by_schedule(_schedule.linear_schedule(1e-2, 1e-4, 10)),
    )

    trace_state, adam_state, decay_state, sched_state = (
        _state_utils.abstract_init(opt, abstract_params)
    )

    # Input weak_type=True is preserved through zeros_like(p) in trace.
    self.assertTrue(trace_state.trace['w'].weak_type)
    self.assertEqual(
        trace_state.trace['w'].sharding, NamedSharding(mesh, P('x', 'y'))
    )
    # Explicit mu_dtype=bfloat16 clears weak_type while preserving sharding.
    self.assertEqual(adam_state.mu['w'].dtype, jnp.bfloat16)
    self.assertFalse(adam_state.mu['w'].weak_type)
    self.assertEqual(adam_state.nu['w'].dtype, jnp.float32)
    self.assertEqual(
        adam_state.mu['w'].sharding, NamedSharding(mesh, P('x', 'y'))
    )
    self.assertEqual(
        adam_state.nu['w'].sharding, NamedSharding(mesh, P('x', 'y'))
    )
    self.assertEqual(adam_state.count.dtype, jnp.int32)
    self.assertEqual(adam_state.count.sharding, NamedSharding(mesh, P()))
    self.assertEqual(decay_state, base.EmptyState())
    self.assertEqual(sched_state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_reduced_axis_adafactor_and_sm3(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    # Shape (8, 2, 16) -> two largest dims are axis 2 (16) and axis 0 (8).
    abstract_params = {
        'kernel': _sharded_sds((8, 2, 16), mesh, P('x', None, 'y')),
    }

    with self.subTest('adafactor'):
      factored_state = _state_utils.abstract_init(
          factorized.scale_by_factored_rms(min_dim_size_to_factor=1),
          abstract_params,
      )
      self.assertEqual(factored_state.v_row['kernel'].shape, (8, 2))
      self.assertEqual(
          factored_state.v_row['kernel'].sharding,
          NamedSharding(mesh, P('x')),
      )
      self.assertEqual(factored_state.v_col['kernel'].shape, (2, 16))
      self.assertEqual(
          factored_state.v_col['kernel'].sharding,
          NamedSharding(mesh, P(None, 'y')),
      )
      self.assertEqual(
          factored_state.v['kernel'].sharding, NamedSharding(mesh, P())
      )
      self.assertEqual(factored_state.count.sharding, NamedSharding(mesh, P()))

    with self.subTest('sm3'):
      sm3_state = _state_utils.abstract_init(
          transform.scale_by_sm3(), abstract_params
      )
      mu_list = sm3_state.mu['kernel']
      self.assertLen(mu_list, 3)
      self.assertEqual([m.shape for m in mu_list], [(8,), (2,), (16,)])
      self.assertEqual(
          [m.sharding for m in mu_list],
          [
              NamedSharding(mesh, P('x')),
              NamedSharding(mesh, P()),
              NamedSharding(mesh, P('y')),
          ],
      )
      self.assertEqual(
          sm3_state.nu['kernel'].sharding,
          NamedSharding(mesh, P('x', None, 'y')),
      )

  def test_abstract_init_expanded_history_lbfgs(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 2, 8), mesh, P('x', None, 'y')),
    }

    lbfgs_state = _state_utils.abstract_init(
        transform.scale_by_lbfgs(memory_size=5), abstract_params
    )

    self.assertEqual(lbfgs_state.diff_params_memory['w'].shape, (5, 4, 2, 8))
    self.assertEqual(
        lbfgs_state.diff_params_memory['w'].sharding,
        NamedSharding(mesh, P(None, 'x', None, 'y')),
    )
    self.assertEqual(
        lbfgs_state.diff_updates_memory['w'].sharding,
        NamedSharding(mesh, P(None, 'x', None, 'y')),
    )
    self.assertEqual(
        lbfgs_state.params['w'].sharding,
        NamedSharding(mesh, P('x', None, 'y')),
    )
    self.assertEqual(lbfgs_state.weights_memory.shape, (5,))
    self.assertEqual(
        lbfgs_state.weights_memory.sharding, NamedSharding(mesh, P())
    )
    self.assertEqual(lbfgs_state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_random_prng_key_state(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 8), mesh, P('x', 'y')),
    }

    noise_state = _state_utils.abstract_init(
        transform.add_noise(1.0, 0.9, key=0), abstract_params
    )

    self.assertIsInstance(noise_state.rng_key, jax.ShapeDtypeStruct)
    self.assertEqual(noise_state.rng_key.sharding, NamedSharding(mesh, P()))
    self.assertEqual(noise_state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_masked_partition_and_injected_hparams(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'matrix': _sharded_sds((4, 8), mesh, P('x', 'y')),
        'bias': _sharded_sds((8,), mesh, P('y')),
    }
    partitioned = combine.partition(
        transforms={
            'dense': transform.trace(decay=0.9),
            'other': transform.scale_by_adam(),
        },
        param_labels={'matrix': 'dense', 'bias': 'other'},
    )
    opt = _inject.inject_hyperparams(
        lambda learning_rate: combine.chain(
            partitioned, transform.scale_by_learning_rate(learning_rate)
        )
    )(learning_rate=_schedule.linear_schedule(1e-2, 1e-4, 10))

    state = _state_utils.abstract_init(opt, abstract_params)
    part_state = state.inner_state[0]
    dense_trace = part_state.inner_states['dense'].inner_state.trace
    other_adam = part_state.inner_states['other'].inner_state

    self.assertEqual(
        dense_trace['matrix'].sharding, NamedSharding(mesh, P('x', 'y'))
    )
    self.assertEqual(
        other_adam.mu['bias'].sharding, NamedSharding(mesh, P('y'))
    )
    self.assertEqual(dense_trace['bias'], wrappers.MaskedNode())
    self.assertEqual(other_adam.mu['matrix'], wrappers.MaskedNode())
    self.assertEqual(
        state.hyperparams['learning_rate'].sharding, NamedSharding(mesh, P())
    )
    self.assertEqual(
        state.hyperparams_states['learning_rate'].count.sharding,
        NamedSharding(mesh, P()),
    )
    self.assertEqual(state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_norm_and_auxiliary_array_in_init(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 8), mesh, P('x', 'y')),
    }

    def custom_init(params):
      sq_norm = sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
      return {
          'init_norm': jnp.sqrt(sq_norm),
          'scaled_params': jax.tree.map(lambda p: 2.0 * p, params),
          'coeffs': jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32),
      }

    state = _state_utils.abstract_init(custom_init, abstract_params)

    self.assertEqual(state['init_norm'].shape, ())
    self.assertEqual(state['init_norm'].sharding, NamedSharding(mesh, P()))
    self.assertEqual(
        state['scaled_params']['w'].sharding,
        NamedSharding(mesh, P('x', 'y')),
    )
    self.assertEqual(state['coeffs'].shape, (3,))
    self.assertEqual(state['coeffs'].sharding, NamedSharding(mesh, P()))

  def test_abstract_init_concrete_array_inputs_return_abstract_sds(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_single_device_auto_mesh()
    sharding = NamedSharding(mesh, P())
    concrete_params = {
        'w': jax.device_put(jnp.ones((4, 8), jnp.float32), sharding),
        'b': jax.device_put(jnp.zeros((8,), jnp.float32), sharding),
    }

    opt_state = _state_utils.abstract_init(alias.adamw(1e-3), concrete_params)

    for leaf in jax.tree.leaves(opt_state):
      self.assertIsInstance(leaf, jax.ShapeDtypeStruct)
      self.assertNotIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.sharding, sharding)

  def test_abstract_init_matches_jit_init_materialization(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_single_device_auto_mesh()
    sharding = NamedSharding(mesh, P())
    abstract_params = {
        'w': jax.ShapeDtypeStruct((4, 8), jnp.float32, sharding=sharding),
        'b': jax.ShapeDtypeStruct((8,), jnp.float32, sharding=sharding),
    }
    concrete_params = {
        'w': jax.device_put(jnp.ones((4, 8), jnp.float32), sharding),
        'b': jax.device_put(jnp.zeros((8,), jnp.float32), sharding),
    }
    opt = _inject.inject_hyperparams(alias.adamw)(learning_rate=1e-3)

    abstract_state = _state_utils.abstract_init(opt, abstract_params)
    out_shardings = jax.tree.map(lambda x: x.sharding, abstract_state)
    materialized = jax.jit(opt.init, out_shardings=out_shardings)(
        concrete_params
    )
    materialized_sds = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape,
            x.dtype,
            sharding=x.sharding,
            weak_type=jax.typeof(x).weak_type,
        ),
        materialized,
    )

    self.assertEqual(abstract_state, materialized_sds)

  def test_abstract_init_outer_mesh_isolation_and_restoration(self):
    self._skip_if_sharding_unsupported()
    inner_mesh = _make_auto_mesh()
    outer_mesh = jax.make_mesh((1,), ('dp',), axis_types=(AxisType.Explicit,))
    abstract_params = {
        'w': _sharded_sds((4, 8), inner_mesh, P('x', 'y')),
    }

    with jax.set_mesh(outer_mesh):
      opt_state = _state_utils.abstract_init(
          transform.scale_by_adam(), abstract_params
      )
      self.assertEqual(jax.sharding.get_mesh(), outer_mesh)
      self.assertEqual(
          jax.sharding.get_abstract_mesh(), outer_mesh.abstract_mesh
      )

      def failing_init(params):
        del params
        raise RuntimeError('boom')

      with self.assertRaisesRegex(RuntimeError, 'boom'):
        _state_utils.abstract_init(failing_init, abstract_params)
      self.assertEqual(jax.sharding.get_mesh(), outer_mesh)
      self.assertEqual(
          jax.sharding.get_abstract_mesh(), outer_mesh.abstract_mesh
      )

    self.assertEqual(
        opt_state.mu['w'].sharding, NamedSharding(inner_mesh, P('x', 'y'))
    )
    self.assertEqual(opt_state.count.sharding, NamedSharding(inner_mesh, P()))

  def test_abstract_init_partially_sharded_and_multi_mesh_error(self):
    self._skip_if_sharding_unsupported()
    mesh_a = _make_auto_mesh()
    mesh_b = jax.sharding.AbstractMesh(
        (4,), ('c',), axis_types=(AxisType.Auto,)
    )

    with self.subTest('partially_sharded_replicates_unsharded_leaves'):
      partial_params = {
          'sharded': _sharded_sds((4, 8), mesh_a, P('x', 'y')),
          'unsharded': jax.ShapeDtypeStruct((4,), jnp.float32),
      }
      state = _state_utils.abstract_init(
          transform.scale_by_adam(), partial_params
      )
      self.assertEqual(
          state.mu['sharded'].sharding, NamedSharding(mesh_a, P('x', 'y'))
      )
      self.assertEqual(
          state.mu['unsharded'].sharding, NamedSharding(mesh_a, P())
      )
      self.assertEqual(state.count.sharding, NamedSharding(mesh_a, P()))

    with self.subTest('raises_on_multiple_distinct_meshes'):
      multi_params = {
          'enc': _sharded_sds((4, 8), mesh_a, P('x', 'y')),
          'dec': _sharded_sds((8,), mesh_b, P('c')),
      }
      with self.assertRaisesRegex(ValueError, 'share the same mesh'):
        _state_utils.abstract_init(transform.scale_by_adam(), multi_params)

  def test_abstract_init_unsharded_single_device_and_empty_tree(self):
    opt = transform.scale_by_adam()

    with self.subTest('unsharded_sds'):
      unsharded = {'w': jax.ShapeDtypeStruct((4, 8), jnp.float32)}
      state = _state_utils.abstract_init(opt, unsharded)
      self.assertIsNone(state.mu['w'].sharding)
      self.assertIsNone(state.count.sharding)

    with self.subTest('single_device_sharding'):
      single_dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
      single_params = {
          'w': jax.ShapeDtypeStruct((4, 8), jnp.float32, sharding=single_dev)
      }
      state = _state_utils.abstract_init(opt, single_params)
      self.assertIsInstance(state.mu['w'], jax.ShapeDtypeStruct)
      self.assertIsNone(state.count.sharding)

    with self.subTest('empty_pytree'):
      state = _state_utils.abstract_init(opt, {})
      self.assertEqual(state.mu, {})
      self.assertEqual(state.nu, {})
      self.assertIsNone(state.count.sharding)

  def test_abstract_init_memory_kind_handling(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()

    with self.subTest('uniform_pinned_host'):
      pinned_params = {
          'w': _sharded_sds(
              (4, 8), mesh, P('x', 'y'), memory_kind='pinned_host'
          )
      }
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, key=0), transform.scale_by_adam()
      )
      noise_state, adam_state = _state_utils.abstract_init(opt, pinned_params)
      self.assertEqual(
          adam_state.mu['w'].sharding,
          NamedSharding(mesh, P('x', 'y'), memory_kind='pinned_host'),
      )
      self.assertEqual(
          adam_state.count.sharding,
          NamedSharding(mesh, P(), memory_kind='pinned_host'),
      )
      self.assertEqual(
          noise_state.rng_key.sharding,
          NamedSharding(mesh, P(), memory_kind='pinned_host'),
      )

    with self.subTest('mixed_pinned_host_and_device_on_same_mesh'):
      mixed_params = {
          'embed': _sharded_sds(
              (4, 8), mesh, P('x', 'y'), memory_kind='pinned_host'
          ),
          'mlp': _sharded_sds((4, 8), mesh, P('x', 'y'), memory_kind='device'),
      }
      state = _state_utils.abstract_init(
          transform.scale_by_adam(), mixed_params
      )
      self.assertEqual(
          state.mu['mlp'].sharding, NamedSharding(mesh, P('x', 'y'))
      )
      self.assertEqual(state.count.sharding, NamedSharding(mesh, P()))

  def test_abstract_init_older_jax_version_raises(self):
    self._skip_if_sharding_unsupported()
    mesh = _make_auto_mesh()
    abstract_params = {
        'w': _sharded_sds((4, 8), mesh, P('x', 'y')),
    }

    self.enterContext(mock.patch.object(jax, '__version__', '0.5.3'))
    with self.assertRaisesRegex(NotImplementedError, 'requires jax>=0.7.2'):
      _state_utils.abstract_init(transform.scale_by_adam(), abstract_params)


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
