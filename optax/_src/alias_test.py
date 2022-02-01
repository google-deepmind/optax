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
from optax._src import update


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
      (
          dict(opt_name='sgd', opt=lambda: alias.sgd(1e-3, 0.9)),
          dict(opt_name='adafactor', opt=lambda: alias.adafactor(5e-3)),
          dict(opt_name='adagrad', opt=lambda: alias.adagrad(1.0)),
          dict(opt_name='adam', opt=lambda: alias.adam(1e-1)),
          dict(opt_name='adamw', opt=lambda: alias.adamw(1e-1)),
          dict(opt_name='lars', opt=lambda: alias.lars(1.0)),
          dict(opt_name='lamb', opt=lambda: alias.lamb(1e-3)),
          dict(
              opt_name='noisy_sgd',
              opt=lambda: alias.noisy_sgd(1e-3, eta=1e-4)),
          dict(opt_name='rmsprop', opt=lambda: alias.rmsprop(5e-3)),
          dict(
              opt_name='rmsprop_momentum',
              opt=lambda: alias.rmsprop(5e-3, momentum=0.9)),
          dict(opt_name='fromage', opt=lambda: alias.fromage(5e-3)),
          dict(opt_name='adabelief', opt=lambda: alias.adabelief(1e-2)),
          dict(opt_name='radam', opt=lambda: alias.radam(5e-3)),
          dict(opt_name='sm3', opt=lambda: alias.sm3(1.0)),
          dict(opt_name='yogi', opt=lambda: alias.yogi(1e-1)),
          dict(
              opt_name='dpsgd',
              opt=lambda: alias.dpsgd(1e-3, 10.0, 0.001, 0, 0.2)),
      ),
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32, jnp.complex64),
  )
  def test_optimization(self, opt_name, opt, target, dtype):
    if (opt_name in ('fromage', 'noisy_sgd', 'sm3') and
        jnp.iscomplexobj(dtype)):
      raise absltest.SkipTest(
          f'{opt_name} does not support complex parameters.')

    opt = opt()
    initial_params, final_params, get_updates = target(dtype)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      if opt_name == 'dpsgd':
        updates = updates[None]
      # Complex gradients need to be conjugated before being added to parameters
      # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
      updates = jax.tree_map(lambda x: x.conj(), updates)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(10000):
      params, state = step(params, state)

    chex.assert_tree_all_close(params, final_params, rtol=3e-2, atol=3e-2)

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
