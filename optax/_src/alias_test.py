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
from optax._src import numerics
from optax._src import schedule
from optax._src import update

_OPTIMIZERS_UNDER_TEST = (
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1e-3, momentum=0.9)),
    dict(opt_name='adafactor', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='adagrad', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adam', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamax', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamaxw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='amsgrad', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='lars', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lamb', opt_kwargs=dict(learning_rate=1e-3)),
    dict(opt_name='noisy_sgd', opt_kwargs=dict(learning_rate=1e-3, eta=1e-4)),
    dict(opt_name='novograd', opt_kwargs=dict(learning_rate=1e-3)),
    dict(
        opt_name='optimistic_gradient_descent',
        opt_kwargs=dict(learning_rate=2e-3, alpha=0.7, beta=0.1)),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=5e-3, momentum=0.9)),
    dict(opt_name='fromage', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='adabelief', opt_kwargs=dict(learning_rate=1e-2)),
    dict(opt_name='radam', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='sm3', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='yogi', opt_kwargs=dict(learning_rate=1e-1)),
    dict(
        opt_name='dpsgd',
        opt_kwargs=dict(
            learning_rate=1e-3,
            l2_norm_clip=10.,
            noise_multiplier=1e-3,
            seed=0,
            momentum=0.2)),
)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  if jnp.iscomplexobj(dtype):
    final_params *= 1 + 1j

  @jax.grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  if jnp.iscomplexobj(dtype):
    a *= 1 + 1j

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return (numerics.abs_sq(a - params[0]) +
            b * numerics.abs_sq(params[1] - params[0]**2))

  return initial_params, final_params, get_updates


class AliasTest(chex.TestCase):

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32, jnp.complex64),
  )
  def test_optimization(self, opt_name, opt_kwargs, target, dtype):
    if (opt_name
        in ('fromage', 'noisy_sgd', 'sm3', 'optimistic_gradient_descent') and
        jnp.iscomplexobj(dtype)):
      raise absltest.SkipTest(
          f'{opt_name} does not support complex parameters.')

    opt = getattr(alias, opt_name)(**opt_kwargs)
    initial_params, final_params, get_updates = target(dtype)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      if opt_name == 'dpsgd':
        updates = updates[None]
      # Complex gradients need to be conjugated before being added to parameters
      # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
      updates = jax.tree_util.tree_map(lambda x: x.conj(), updates)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(10000):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @chex.all_variants
  @parameterized.product(_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/deepmind/optax/issues/412.
    opt_factory = getattr(alias, opt_name)
    opt = opt_factory(**opt_kwargs)
    if opt_name == 'adafactor':
      # Adafactor wrapped in inject_hyperparams currently needs a static
      # argument to be specified in order to be jittable. See issue
      # https://github.com/deepmind/optax/issues/412.
      opt_inject = schedule.inject_hyperparams(
          opt_factory, static_args=('min_dim_size_to_factor',))(**opt_kwargs)
    else:
      opt_inject = schedule.inject_hyperparams(opt_factory)(**opt_kwargs)

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

  @parameterized.named_parameters([
      ('float32', 'float32'),
      ('bfloat16', 'bfloat16'),
      ('complex64', 'complex64'),
      ('None', None),
  ])
  def test_explicit_dtype(self, dtype):
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)  # None -> float32
    tx = alias.sgd(0.1, momentum=0.9, accumulator_dtype=dtype)
    trace_state, _ = tx.init(jnp.array([0.0, 0.0]))
    self.assertEqual(expected_dtype, trace_state.trace.dtype)
    tx = alias.adam(0.1, mu_dtype=dtype)
    adam_state, _ = tx.init(jnp.array([0.0, 0.0]))
    self.assertEqual(expected_dtype, adam_state.mu.dtype)
    tx = alias.adamw(0.1, mu_dtype=dtype)
    adam_state, _, _ = tx.init(jnp.array([0.0, 0.0]))
    self.assertEqual(expected_dtype, adam_state.mu.dtype)


if __name__ == '__main__':
  absltest.main()
