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
"""Tests for `cocob.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

from optax import contrib
from optax._src import numerics
from optax._src import update
from optax.schedules import _inject
from optax.tree_utils import _state_utils


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return (numerics.abs_sq(a - params[0]) +
            b * numerics.abs_sq(params[1] - params[0]**2))

  return initial_params, final_params, get_updates


class AliasTest(chex.TestCase):

  @parameterized.product(
      opt_name=('cocob',),
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32,),
  )
  def test_optimization(self, opt_name, target, dtype):

    opt = getattr(contrib, opt_name)()
    initial_params, final_params, get_updates = target(dtype)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    # A no-op change, to verify that tree map works.
    state = _state_utils.tree_map_params(opt, lambda v: v, state)

    for _ in range(10000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @chex.all_variants
  @parameterized.product(opt_name=('cocob',),
                         opt_kwargs=(dict(alpha=100, eps=1e-8),))
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/deepmind/optax/issues/412.
    opt_factory = getattr(contrib, opt_name)
    opt = opt_factory(**opt_kwargs)
    opt_inject = _inject.inject_hyperparams(opt_factory)(**opt_kwargs)

    params = [-jnp.ones((2, 3)), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), -jnp.ones((2, 5, 2))]

    state = self.variant(opt.init)(params)
    updates, new_state = self.variant(opt.update)(grads, state, params)

    state_inject = self.variant(opt_inject.init)(params)
    updates_inject, new_state_inject = self.variant(opt_inject.update)(
        grads, state_inject, params)

    with self.subTest('Equality of updates.'):
      chex.assert_trees_all_close(updates_inject, updates, rtol=1e-4)
    with self.subTest('Equality of new optimizer states.'):
      chex.assert_trees_all_close(
          new_state_inject.inner_state, new_state, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
