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
"""Tests for `sam.py`."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import update
from optax.tree_utils import _state_utils


# TODO(harshm): make LARS and Fromage work with SAM.
_OPTIMIZERS_UNDER_TEST = (
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adam', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamax', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamaxw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='amsgrad', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lamb', opt_kwargs=dict(learning_rate=1.0)),
    dict(
        opt_name='lion',
        opt_kwargs=dict(learning_rate=1.0, b1=0.99),
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


def _setup_mixture(dtype):
  initial_params = jnp.array([-0.4, -0.4], dtype=dtype)
  final_params = jnp.array([2.0, 0.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    x, y = params
    return -jnp.exp(-((x - 2) ** 2) - y**2) - 1.0 * jnp.exp(
        -((x) ** 2 + (y) ** 2 * 100)
    )

  return initial_params, final_params, get_updates


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


class TestOptimizerState(NamedTuple):
  """Inner optimizer state for the SAM tests."""

  aggregate_grads: base.Params


def _test_optimizer(step_size: float) -> base.GradientTransformation:
  """Inner optimizer for the SAM tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behaviour of SAM can be tested.
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


class SAMTest(chex.TestCase):

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      sync_period=(2,),
      target=(_setup_parabola,),
      dtype=(jnp.float32,),
  )
  def test_optimization(self, opt_name, opt_kwargs, sync_period, target, dtype):
    opt = alias.sgd(0.003)
    adv_opt = combine.chain(
        contrib.normalize(), getattr(alias, opt_name)(**opt_kwargs)
    )
    opt = contrib.sam(opt, adv_opt, sync_period=sync_period)
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

    for _ in range(25000 * sync_period):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      sync_period=(2,),
      target=(_setup_parabola,),
      dtype=(jnp.float32,),
  )
  def test_opaque_optimization(
      self, opt_name, opt_kwargs, sync_period, target, dtype
  ):
    base_opt = alias.sgd(0.003)
    adv_opt = combine.chain(
        contrib.normalize(), getattr(alias, opt_name)(**opt_kwargs)
    )
    opt = contrib.sam(
        base_opt, adv_opt, sync_period=sync_period, opaque_mode=True
    )
    initial_params, final_params, get_updates = target(dtype)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(
          updates, state, params, grad_fn=lambda p, _: get_updates(p)
      )
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    # A no-op change, to verify that tree map works.
    state = _state_utils.tree_map_params(opt, lambda v: v, state)

    for _ in range(25000 * sync_period):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)


if __name__ == '__main__':
  absltest.main()
