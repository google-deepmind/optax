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
"""Tests for `mechanic.py`."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import numerics
from optax._src import state_utils
from optax._src import update
from optax._src.contrib import mechanic


# TODO(harshm): make LARS and Fromage work with mechanic.
_OPTIMIZERS_UNDER_TEST = (
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1.0, momentum=0.9)),
    dict(opt_name='adam', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamax', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamaxw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='amsgrad', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lamb', opt_kwargs=dict(learning_rate=1.0)),
    dict(
        opt_name='lion', opt_kwargs=dict(learning_rate=1.0, b1=0.99),
    ),
    dict(opt_name='noisy_sgd', opt_kwargs=dict(learning_rate=1.0, eta=1e-4)),
    dict(opt_name='novograd', opt_kwargs=dict(learning_rate=1.0)),
    dict(
        opt_name='optimistic_gradient_descent',
        opt_kwargs=dict(learning_rate=1.0, alpha=0.7, beta=0.1),
    ),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=1.0, momentum=0.9)),
    dict(opt_name='adabelief', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='radam', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='sm3', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='yogi', opt_kwargs=dict(learning_rate=1.0, b1=0.99)),
)


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


class TestOptimizerState(NamedTuple):
  """Inner optimizer state for the Mechanic tests."""
  aggregate_grads: base.Params


def _test_optimizer(step_size: float) -> base.GradientTransformation:
  """Inner optimizer for the Mechanic tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behaviour of lookahead can be tested.
  def init_fn(params):
    aggregate_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    return TestOptimizerState(aggregate_grads)

  def update_fn(updates, state, params):
    # The test optimizer does not use the parameters, but we check that they
    # have been passed correctly.
    chex.assert_trees_all_equal_shapes(updates, params)
    aggregate_grads = update.apply_updates(state.aggregate_grads, updates)
    updates = jax.tree_util.tree_map(lambda u: step_size * u, updates)
    return updates, TestOptimizerState(aggregate_grads)

  return base.GradientTransformation(init_fn, update_fn)


class MechanicTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.tree_a = (rng.randn(20, 10), rng.randn(20))
    self.tree_b = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict = (1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    self.tree_b_dict = (1.0, {'k1': 2.0, 'k2': (3.0, 4.0)}, 5.0)

    self.array_a = rng.randn(20)
    self.array_b = rng.randn(20)

    self.grads = {'x': np.array(2.), 'y': np.array(-2.)}
    self.initial_params = {'x': np.array(3.), 'y': np.array(-3.)}

  def loop(self, optimizer, num_steps, params):
    """Performs a given number of optimizer steps."""
    init_fn, update_fn = optimizer
    # Use the chex variant to check various function versions (jit, pmap, etc).
    step = self.variant(update_fn)
    opt_state = self.variant(init_fn)(params)

    # A no-op change, to verify that tree map works.
    opt_state = state_utils.tree_map_params(init_fn, lambda v: v, opt_state)

    for _ in range(num_steps):
      updates, opt_state = step(self.grads, opt_state, params)
      print(updates)
      params = update.apply_updates(params, updates)

    return params, opt_state

  @chex.all_variants(with_pmap=False)
  def test_mechanized(self):
    params = self.initial_params
    num_betas = 6

    inner_optimizer = _test_optimizer(-0.1)
    optimizer = mechanic.mechanize(
        inner_optimizer,
        weight_decay=1e-2,
        eps=1e-10,
        s_init=1e-8,
        num_betas=num_betas,
    )

    final_params, final_state = self.loop(
        optimizer=optimizer, num_steps=1, params=params
    )
    expected_m = np.array([1.0e-10] * num_betas)
    expected_v = np.array([0.0] * num_betas)
    expected_s = np.array([1.6666667e-09] * num_betas)

    chex.assert_trees_all_close(expected_m, final_state.m)
    chex.assert_trees_all_close(expected_v, final_state.v)
    chex.assert_trees_all_close(expected_s, final_state.s)
    chex.assert_trees_all_close(final_params, params)
    chex.assert_tree_all_finite((final_params, final_state))

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32,),
  )
  def test_optimization(self, opt_name, opt_kwargs, target, dtype):

    opt = getattr(alias, opt_name)(**opt_kwargs)
    opt = mechanic.mechanize(opt, weight_decay=0.0)
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
    state = state_utils.tree_map_params(opt, lambda v: v, state)

    for _ in range(25000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)


if __name__ == '__main__':
  absltest.main()
