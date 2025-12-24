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

"""Module for testing sharding and related behavior of the optax public API."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import optax
from optax._src import test_utils
from optax.experimental import microbatching


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

OPTIMIZERS = {
    'adam': optax.adam(1.0),
    'sgd': optax.sgd(1.0),
    'adabelief': optax.adabelief(1.0),
    'adamax': optax.adamax(1.0),
    'adagrad': optax.adagrad(1.0),
    'adamw': optax.adamw(1.0),
    'rmsprop': optax.rmsprop(1.0),
    # TODO(mckennar): try to incorporate linesearch into the test.
    'lbfgs': optax.lbfgs(1.0, linesearch=None),
    'adadelta': optax.adadelta(1.0),
    'adafactor': optax.adafactor(),
    'adafactor2': optax.adafactor(min_dim_size_to_factor=1),
    'adan': optax.adan(1.0),
    'adamaxw': optax.adamaxw(1.0),
    'amsgrad': optax.amsgrad(1.0),
    'fromage': optax.fromage(1.0),
    'lamb': optax.lamb(1.0),
    'lars': optax.lars(1.0),
    'lion': optax.lion(1.0),
    'nadam': optax.nadam(1.0),
    'nadamw': optax.nadamw(1.0),
    'noisy_sgd': optax.noisy_sgd(1.0),
    'novograd': optax.novograd(1.0),
    'optimistic_gradient_descent': optax.optimistic_gradient_descent(1.0),
    'radam': optax.radam(1.0),
    'sm3': optax.sm3(1.0),
    'yogi': optax.yogi(1.0),
}


class ShardingTest(parameterized.TestCase):

  @parameterized.named_parameters(OPTIMIZERS.items())
  def test_init_with_abstract_input(self, optimizer):
    params = jax.ShapeDtypeStruct(shape=(2, 4, 8), dtype=jnp.float32)
    state = optimizer.init(params)
    self.assertIsNotNone(state)

  @parameterized.named_parameters(OPTIMIZERS.items())
  def test_state_sharding_type_init_match_update(self, optimizer):
    if jax.__version__ < '0.7.2':
      self.skipTest('Skipping sharding-in-types test')
    mesh = jax.make_mesh(
        (8,), ('x',), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.P(None, 'x'))

    with jax.set_mesh(mesh):
      params = jnp.zeros((2, 8, 4), dtype=jnp.float16, out_sharding=sharding)

      state0 = optimizer.init(params)
      _, state1 = optimizer.update(params, state0, params)

      type0 = jax.tree.map(jax.typeof, state0)
      type1 = jax.tree.map(jax.typeof, state1)
      test_utils.assert_trees_all_equal(type0, type1)

  @parameterized.named_parameters(OPTIMIZERS.items())
  def test_state_sharding_type_preserved_with_jit(self, optimizer):
    if jax.__version__ < '0.7.2':
      self.skipTest('Skipping sharding-in-types test')
    mesh = jax.make_mesh(
        (8,), ('x',), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.P(None, 'x'))

    with jax.set_mesh(mesh):
      params = jnp.zeros((2, 8, 4), dtype=jnp.float16, out_sharding=sharding)

      state0 = optimizer.init(params)
      state1 = jax.jit(optimizer.init)(params)
      type0 = jax.tree.map(jax.typeof, state0)
      type1 = jax.tree.map(jax.typeof, state1)
      test_utils.assert_trees_all_equal(type0, type1)

      _, state2 = optimizer.update(params, state0, params)
      _, state3 = jax.jit(optimizer.update)(params, state0, params)
      type2 = jax.tree.map(jax.typeof, state2)
      type3 = jax.tree.map(jax.typeof, state3)
      test_utils.assert_trees_all_equal(type2, type3)

  @parameterized.named_parameters(
      ('replicated', jax.sharding.PartitionSpec()),
      ('sharded', jax.sharding.PartitionSpec('x'))
  )
  def test_microbatch_with_explicit_sharding(self, spec):
    if jax.__version__ < '0.8.1':
      self.skipTest('Skipping sharding-in-types test.')
    mesh = jax.make_mesh(
        (8,), ('x',), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      sharding = jax.sharding.NamedSharding(mesh, spec)

      fun = lambda x: (x + 1, jnp.sum(3 * x))
      data = jnp.arange(16, out_sharding=sharding)

      strategy = (
          microbatching.AccumulationType.CONCAT,
          microbatching.AccumulationType.SUM,
      )
      microbatched_fun = microbatching.microbatch(
          fun, argnums=0, microbatch_size=8, accumulator=strategy
      )

      actual = microbatched_fun(data)
      expected = fun(data)

      test_utils.assert_trees_all_equal(
          jax.tree.map(jax.typeof, actual), jax.tree.map(jax.typeof, expected)
      )
      test_utils.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
