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
"""Tests of equivalence between optax and other optimiser libraries."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from jax.experimental import optimizers
import jax.numpy as jnp

from optax._src import alias
from optax._src import update


STEPS = 50
LR = 1e-2
LR_SCHED = lambda _: LR  # Trivial constant "schedule".


class ExperimentalOptimizersEquivalenceTest(chex.TestCase):

  def setUp(self):
    super(ExperimentalOptimizersEquivalenceTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants()
  @parameterized.named_parameters(
      ('sgd', alias.sgd(LR, 0.0),
       optimizers.sgd(LR), 1e-5),
      ('adam', alias.adam(LR, 0.9, 0.999, 1e-8),
       optimizers.adam(LR, 0.9, 0.999), 1e-4),
      ('rmsprop', alias.rmsprop(LR, .9, 0.1),
       optimizers.rmsprop(LR, .9, 0.1), 1e-5),
      ('adagrad', alias.adagrad(LR, 0., 0.,),
       optimizers.adagrad(LR, 0.), 1e-5),
      ('sgd', alias.sgd(LR_SCHED, 0.0),
       optimizers.sgd(LR), 1e-5),
      ('adam', alias.adam(LR_SCHED, 0.9, 0.999, 1e-8),
       optimizers.adam(LR, 0.9, 0.999), 1e-4),
      ('rmsprop', alias.rmsprop(LR_SCHED, .9, 0.1),
       optimizers.rmsprop(LR, .9, 0.1), 1e-5),
      ('adagrad', alias.adagrad(LR_SCHED, 0., 0.,),
       optimizers.adagrad(LR, 0.), 1e-5),
  )
  def test_jax_optimizer_equivalent(self, optax_optimizer, jax_optimizer, rtol):

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = jax_optimizer
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # optax
    optax_params = self.init_params
    state = optax_optimizer.init(optax_params)

    @self.variant
    def step(updates, state):
      return optax_optimizer.update(updates, state)

    for _ in range(STEPS):
      updates, state = step(self.per_step_updates, state)
      optax_params = update.apply_updates(optax_params, updates)

    # Check equivalence.
    chex.assert_tree_all_close(jax_params, optax_params, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
