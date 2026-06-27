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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import optax
from optax._src import test_utils
from optax._src import utils

# Set device count before the JAX backend is initialized so that
# jax.make_mesh((8,), ...) works in the tests below.
jax.config.update('jax_num_cpu_devices', 8)

OPTIMIZERS = {
    'acprop': optax.contrib.acprop(1e-3),
    'ademamix': optax.contrib.ademamix(1e-3),
    'adopt': optax.contrib.adopt(1e-2),
    'cocob': optax.contrib.cocob(),
    'dadapt_adamw': optax.contrib.dadapt_adamw(1e-1),
    'dog': optax.contrib.dog(1.0),
    'dowg': optax.contrib.dowg(1.0),
    'galore': optax.contrib.galore(1e-2, rank=4),
    'madgrad': optax.contrib.madgrad(1e-2),
    'muon': optax.contrib.muon(1e-2),
    'prodigy': optax.contrib.prodigy(1e-1),
    'schedule_free_adamw': optax.contrib.schedule_free_adamw(
        1e-2, warmup_steps=5000
    ),
    'schedule_free_sgd': optax.contrib.schedule_free_sgd(
        1e-2, warmup_steps=5000
    ),
    'simplified_ademamix': optax.contrib.simplified_ademamix(1e-3),
}


class ContribShardingTest(parameterized.TestCase):

  @parameterized.named_parameters(OPTIMIZERS.items())
  def test_state_sharding_type_stable(self, optimizer):
    if utils.parse_version(jax.__version__) < utils.parse_version('0.7.2'):
      self.skipTest('Skipping sharding-in-types test')

    mesh = jax.make_mesh(
        (8,), ('x',), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.P(None, 'x'))

    with jax.set_mesh(mesh):
      params = jnp.zeros(
          (2, 8, 4), dtype=jnp.float16, out_sharding=sharding
      )

      # Eager init and update should have matching sharding types.
      state_eager = optimizer.init(params)
      _, state_after_update = optimizer.update(params, state_eager, params)
      test_utils.assert_trees_all_equal(
          jax.tree.map(jax.typeof, state_eager),
          jax.tree.map(jax.typeof, state_after_update),
      )

      # JIT-compiled init and update should match their eager counterparts.
      state_jit = jax.jit(optimizer.init)(params)
      test_utils.assert_trees_all_equal(
          jax.tree.map(jax.typeof, state_eager),
          jax.tree.map(jax.typeof, state_jit),
      )

      _, state_update_jit = jax.jit(optimizer.update)(
          params, state_eager, params
      )
      test_utils.assert_trees_all_equal(
          jax.tree.map(jax.typeof, state_after_update),
          jax.tree.map(jax.typeof, state_update_jit),
      )


if __name__ == '__main__':
  absltest.main()
