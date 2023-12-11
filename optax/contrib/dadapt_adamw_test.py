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
"""Tests for `dadapt_adamw.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import numerics
from optax._src import update
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
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  return initial_params, final_params, get_updates


class DAdaptAdamWTest(chex.TestCase):
  """Tests for D Adaptation optimizer."""

  @parameterized.product(
      opt_name=('dadapt_adamw',),
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

    for _ in range(15000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
  absltest.main()
