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
from flax import optim
from jax.example_libraries import optimizers
import jax.numpy as jnp

from optax._src import alias
from optax._src import update


STEPS = 50
LR = 1e-2
LR_SCHED = lambda _: LR  # Trivial constant "schedule".


class OptimizersEquivalenceTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4., 5.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3., 1.]))

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', alias.sgd(LR, 0.0),
       optimizers.sgd(LR), 1e-5),
      ('adam', alias.adam(LR, 0.9, 0.999, 1e-8),
       optimizers.adam(LR, 0.9, 0.999), 1e-4),
      ('rmsprop', alias.rmsprop(LR, decay=.9, eps=0.1),
       optimizers.rmsprop(LR, .9, 0.1), 1e-5),
      ('rmsprop_momentum', alias.rmsprop(
          LR, decay=.9, eps=0.1, momentum=0.9),
       optimizers.rmsprop_momentum(LR, .9, 0.1, 0.9), 1e-5),
      ('adagrad', alias.adagrad(LR, 0., 0.,),
       optimizers.adagrad(LR, 0.), 1e-5),
      ('sgd', alias.sgd(LR_SCHED, 0.0),
       optimizers.sgd(LR), 1e-5),
      ('adam', alias.adam(LR_SCHED, 0.9, 0.999, 1e-8),
       optimizers.adam(LR, 0.9, 0.999), 1e-4),
      ('rmsprop', alias.rmsprop(LR_SCHED, decay=.9, eps=0.1),
       optimizers.rmsprop(LR, .9, 0.1), 1e-5),
      ('rmsprop_momentum', alias.rmsprop(
          LR_SCHED, decay=.9, eps=0.1, momentum=0.9),
       optimizers.rmsprop_momentum(LR, .9, 0.1, 0.9), 1e-5),
      ('adagrad', alias.adagrad(LR_SCHED, 0., 0.,),
       optimizers.adagrad(LR, 0.), 1e-5),
      ('sm3', alias.sm3(LR), optimizers.sm3(LR), 1e-2),
  )
  def test_jax_optimizer_equivalent(self, optax_optimizer, jax_optimizer, rtol):

    # example_libraries/optimizers.py
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


class FlaxOptimizersEquivalenceTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (
        jnp.array([1., 0.1, 1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (
        jnp.array([0., 0.3, 500., 5.]), jnp.array([300., 3.]))

  @parameterized.named_parameters(
      ('sgd',
       alias.sgd(LR),
       optim.GradientDescent(LR)),
      ('momentum',
       alias.sgd(LR, momentum=0.9),
       optim.Momentum(LR, beta=0.9)),  # Different names.
      ('nesterov_momentum',
       alias.sgd(LR, momentum=0.9, nesterov=True),
       optim.Momentum(LR, beta=0.9, nesterov=True)),
      ('rmsprop',
       alias.rmsprop(LR),
       optim.RMSProp(LR)),
      ('centered_rmsprop',
       alias.rmsprop(LR, centered=True),
       optim.RMSProp(LR, centered=True)),
      ('adam',
       alias.adam(LR),
       optim.Adam(LR)),
      ('adam_w',
       alias.adamw(LR, weight_decay=1e-4),
       optim.Adam(LR, weight_decay=1e-4)),  # Different name.
      ('adagrad',
       alias.adagrad(LR, initial_accumulator_value=0.),  # Different default!
       optim.Adagrad(LR)),
      ('lamb',
       alias.lamb(LR),
       optim.LAMB(LR)),
      ('lars',
       alias.lars(
           LR, weight_decay=.5, trust_coefficient=0.003,
           momentum=0.9, eps=1e-3),
       optim.LARS(
           LR, weight_decay=.5, trust_coefficient=0.003,
           beta=0.9, eps=1e-3)),
      ('adafactor',
       alias.adafactor(
           learning_rate=LR / 10.,
           factored=True,
           multiply_by_parameter_scale=True,
           clipping_threshold=1.0,
           decay_rate=0.8,
           min_dim_size_to_factor=2),
       optim.Adafactor(
           learning_rate=LR / 10.,
           factored=True,
           multiply_by_parameter_scale=True,
           clipping_threshold=1.0,
           decay_rate=0.8,
           min_dim_size_to_factor=2)),
  )
  def test_flax_optim_equivalence(self, optax_optimizer, flax_optimizer):

    # flax/optim
    flax_params = self.init_params
    flax_optimizer = flax_optimizer.create(flax_params)
    for _ in range(STEPS):
      flax_optimizer = flax_optimizer.apply_gradient(
          self.per_step_updates)
      flax_params = flax_optimizer.target

    # optax
    optax_params = self.init_params
    state = optax_optimizer.init(optax_params)
    for _ in range(STEPS):
      updates, state = optax_optimizer.update(
          self.per_step_updates, state, optax_params)
      optax_params = update.apply_updates(optax_params, updates)

    # Check equivalence.
    chex.assert_tree_all_close(flax_params, optax_params, rtol=2e-4)


if __name__ == '__main__':
  absltest.main()
