# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Sharding and dtype-stability tests for optax.contrib optimizers."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import optax
from optax._src import test_utils
from optax._src import utils


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

# Optimizers that support initialization from abstract inputs
# (jax.ShapeDtypeStruct). Optimizers that require concrete parameter values
# during init (e.g. to record an initial norm) are excluded from this dict
# but tested in SHARDING_OPTIMIZERS below.
ABSTRACT_INIT_OPTIMIZERS = {
    'acprop': optax.contrib.acprop(1e-3),
    'ademamix': optax.contrib.ademamix(1e-3),
    'adopt': optax.contrib.adopt(1e-2),
    'cocob': optax.contrib.cocob(),
    'galore': optax.contrib.galore(1e-2, rank=4),
    'madgrad': optax.contrib.madgrad(1e-2),
    'muon': optax.contrib.muon(1e-2),
    'simplified_ademamix': optax.contrib.simplified_ademamix(1e-3),
}

# All optimizers tested for sharding-type stability. Some (dadapt_adamw, dog,
# dowg, prodigy, schedule_free_*) require concrete params at init and are
# therefore omitted from ABSTRACT_INIT_OPTIMIZERS above.
SHARDING_OPTIMIZERS = {
    **ABSTRACT_INIT_OPTIMIZERS,
    'dadapt_adamw': optax.contrib.dadapt_adamw(1e-1),
    'dog': optax.contrib.dog(1.0),
    'dowg': optax.contrib.dowg(1.0),
    'prodigy': optax.contrib.prodigy(1e-1),
    'schedule_free_adamw': optax.contrib.schedule_free_adamw(
        1e-2, warmup_steps=5000
    ),
    'schedule_free_sgd': optax.contrib.schedule_free_sgd(
        1e-2, warmup_steps=5000
    ),
}


class ContribShardingTest(parameterized.TestCase):

  @parameterized.named_parameters(ABSTRACT_INIT_OPTIMIZERS.items())
  def test_init_with_abstract_input(self, optimizer):
    params = jax.ShapeDtypeStruct(shape=(2, 4, 8), dtype=jnp.float32)
    state = optimizer.init(params)
    self.assertIsNotNone(state)

  @parameterized.named_parameters(SHARDING_OPTIMIZERS.items())
  def test_state_sharding_type_init_match_update(self, optimizer):
    if utils.parse_version(jax.__version__) < utils.parse_version('0.7.2'):
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

  @parameterized.named_parameters(SHARDING_OPTIMIZERS.items())
  def test_state_sharding_type_preserved_with_jit(self, optimizer):
    if utils.parse_version(jax.__version__) < utils.parse_version('0.7.2'):
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


if __name__ == '__main__':
  absltest.main()
