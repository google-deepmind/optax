# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
from optax._src import numerics
from optax.experimental import gradient_solver


_GRAD_TRANSFORMS_UNDER_TEST = (
    dict(gt_name='sgd', gt_kwargs=dict(learning_rate=1e-3, momentum=0.9)),
    dict(gt_name='adafactor', gt_kwargs=dict(learning_rate=5e-3)),
    dict(gt_name='adagrad', gt_kwargs=dict(learning_rate=1.0)),
    dict(gt_name='adam', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='adamw', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='adamax', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='adamaxw', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='amsgrad', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='lars', gt_kwargs=dict(learning_rate=1.0)),
    dict(gt_name='lamb', gt_kwargs=dict(learning_rate=1e-3)),
    dict(
        gt_name='lion', gt_kwargs=dict(learning_rate=1e-2, weight_decay=1e-4),
    ),
    dict(gt_name='nadam', gt_kwargs=dict(learning_rate=1e-2)),
    dict(gt_name='nadamw', gt_kwargs=dict(learning_rate=1e-2)),
    dict(gt_name='noisy_sgd', gt_kwargs=dict(learning_rate=1e-3, eta=1e-4)),
    dict(gt_name='novograd', gt_kwargs=dict(learning_rate=1e-3)),
    dict(
        gt_name='optimistic_gradient_descent',
        gt_kwargs=dict(learning_rate=2e-3, alpha=0.7, beta=0.1),
    ),
    dict(gt_name='rmsprop', gt_kwargs=dict(learning_rate=5e-3)),
    dict(gt_name='rmsprop', gt_kwargs=dict(learning_rate=5e-3, momentum=0.9)),
    dict(gt_name='fromage', gt_kwargs=dict(learning_rate=5e-3)),
    dict(gt_name='adabelief', gt_kwargs=dict(learning_rate=1e-2)),
    dict(gt_name='radam', gt_kwargs=dict(learning_rate=5e-3)),
    dict(gt_name='rprop', gt_kwargs=dict(learning_rate=1e-1)),
    dict(gt_name='sm3', gt_kwargs=dict(learning_rate=1.0)),
    dict(gt_name='yogi', gt_kwargs=dict(learning_rate=1e-1)),
)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  def obj_fun(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, obj_fun


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  def obj_fun(params):
    return (numerics.abs_sq(a - params[0]) +
            b * numerics.abs_sq(params[1] - params[0]**2))

  return initial_params, final_params, obj_fun


class SolverWrapperTest(chex.TestCase):

  @parameterized.product(
      _GRAD_TRANSFORMS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32,),
  )
  def test_optimization(self, gt_name, gt_kwargs, target, dtype):
    opt = getattr(alias, gt_name)(**gt_kwargs)
    initial_params, final_params, obj_fun = target(dtype)

    init, step = gradient_solver.gradient_solver(obj_fun, opt)

    params = initial_params
    state = init(params)
    step = jax.jit(step)
    for _ in range(10_000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

if __name__ == '__main__':
  absltest.main()
