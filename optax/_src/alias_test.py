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
"""Tests for `alias.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
from jax.experimental import optimizers
import jax.numpy as jnp

from optax._src import alias
from optax._src import transform
from optax._src import update


STEPS = 50
LR = 1e-2


class AliasTest(chex.TestCase):

  def setUp(self):
    super(AliasTest, self).setUp()
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

  @parameterized.named_parameters(
      ('sgd', alias.sgd(1e-2, 0.0)),
      ('adam', alias.adam(1e-1)),
      ('adamw', alias.adamw(1e-1)),
      ('lamb', alias.adamw(1e-1)),
      ('rmsprop', alias.rmsprop(1e-1)),
      ('fromage', transform.scale_by_fromage(-1e-2)),
      ('adabelief', alias.adabelief(1e-1)),
  )
  def test_parabel(self, opt):
    initial_params = jnp.array([-1.0, 10.0, 1.0])
    final_params = jnp.array([1.0, -1.0, 1.0])

    @jax.grad
    def get_updates(params):
      return jnp.sum((params - final_params)**2)

    @jax.jit
    def step(params, state):
      updates, state = opt.update(get_updates(params), state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(1000):
      params, state = step(params, state)

    chex.assert_tree_all_close(params, final_params, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      ('sgd', alias.sgd(2e-3, 0.2)),
      ('adam', alias.adam(1e-1)),
      ('adamw', alias.adamw(1e-1)),
      ('lamb', alias.adamw(1e-1)),
      ('rmsprop', alias.rmsprop(5e-3)),
      ('fromage', transform.scale_by_fromage(-5e-3)),
      ('adabelief', alias.adabelief(1e-1)),
  )
  def test_rosenbrock(self, opt):
    a = 1.0
    b = 100.0
    initial_params = jnp.array([0.0, 0.0])
    final_params = jnp.array([a, a**2])

    @jax.grad
    def get_updates(params):
      return (a - params[0])**2 + b * (params[1] - params[0]**2)**2

    @jax.jit
    def step(params, state):
      updates, state = opt.update(get_updates(params), state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(10000):
      params, state = step(params, state)

    chex.assert_tree_all_close(params, final_params, rtol=3e-2, atol=3e-2)


if __name__ == '__main__':
  absltest.main()
