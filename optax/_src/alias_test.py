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
import jax.numpy as jnp

from optax._src import alias
from optax._src import update


class AliasTest(chex.TestCase):

  @parameterized.named_parameters(
      ('sgd', lambda: alias.sgd(1e-2, 0.0)),
      ('adam', lambda: alias.adam(1e-1)),
      ('adamw', lambda: alias.adamw(1e-1)),
      ('lamb', lambda: alias.adamw(1e-1)),
      ('rmsprop', lambda: alias.rmsprop(1e-1)),
      ('fromage', lambda: alias.fromage(1e-2)),
      ('adabelief', lambda: alias.adabelief(1e-1)),
      ('radam', lambda: alias.radam(1e-1)),
      ('yogi', lambda: alias.yogi(1.0)),
  )
  def test_parabel(self, opt):
    opt = opt()

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
      ('sgd', lambda: alias.sgd(2e-3, 0.2)),
      ('adam', lambda: alias.adam(1e-1)),
      ('adamw', lambda: alias.adamw(1e-1)),
      ('lamb', lambda: alias.adamw(1e-1)),
      ('rmsprop', lambda: alias.rmsprop(5e-3)),
      ('fromage', lambda: alias.fromage(5e-3)),
      ('adabelief', lambda: alias.adabelief(1e-1)),
      ('radam', lambda: alias.radam(1e-3)),
      ('yogi', lambda: alias.yogi(1.0)),
  )
  def test_rosenbrock(self, opt):
    opt = opt()

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
