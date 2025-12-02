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
"""Tests for methods in `inject.py`."""

import functools
from typing import NamedTuple, Union

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from optax import schedules
from optax import transforms

from optax._src import base
from optax._src import test_utils
from optax._src import transform
from optax._src import wrappers
import optax.tree


class ExampleState(NamedTuple):
  total: jax.typing.ArrayLike


class ExampleStatefulSchedule(base.StatefulSchedule):

  def init(self) -> ExampleState:
    return ExampleState(total=jnp.zeros([], dtype=jnp.int32))

  def update(self, state: ExampleState, **extra_args) -> ExampleState:
    total = state.total + extra_args['addendum']
    return ExampleState(total=total)

  def __call__(self, state: ExampleState, **extra_args) -> jax.typing.ArrayLike:
    return state.total


class InjectHyperparamsTest(parameterized.TestCase):
  """Tests for the inject_hyperparams wrapper."""

  def test_updates(self):
    optim = schedules.inject_hyperparams(transform.scale)(  # stateless
        step_size=schedules.piecewise_constant_schedule(
            3.0, {1: 5, 7: 2, 12: 1.5}
        )
    )

    params = [jnp.zeros([], dtype=jnp.float32)]
    state = jax.jit(optim.init)(params)

    # A no-op change, to verify that tree map works.
    state = optax.tree.map_params(optim, lambda v: v, state)

    update_fn = jax.jit(optim.update)
    expected_step_size = [3.0] * 2 + [15.0] * 6 + [30.0] * 5 + [45.0] * 3

    grads = [jnp.ones([], dtype=jnp.float32)]
    for i in range(15):
      updates, state = update_fn(grads, state, params=params)
      np.testing.assert_almost_equal(updates[0], expected_step_size[i + 1])

  def test_hyperparams_state(self):
    optim = schedules.inject_hyperparams(transform.trace)(  # stateful
        decay=schedules.piecewise_constant_schedule(0.8, {3: 0.5, 9: 1.25}),
        nesterov=True,
    )

    params = [jnp.zeros([2, 3]) for _ in range(3)]
    state = jax.jit(optim.init)(params)
    update_fn = jax.jit(optim.update)

    expected_mom = [0.8] * 4 + [0.4] * 6 + [0.5] * 2
    grads = jax.tree.map(jnp.ones_like, params)
    for i in range(12):
      np.testing.assert_almost_equal(
          state.hyperparams['decay'], expected_mom[i]
      )
      _, state = update_fn(grads, state)

    np.testing.assert_almost_equal(state.hyperparams['decay'], expected_mom[-1])

  def test_constant_hyperparams(self):
    optim = schedules.inject_hyperparams(
        transform.scale_by_adam
    )(b1=0.0, b2=0.0)

    params = [jnp.zeros([2, 3]) for _ in range(3)]
    state = jax.jit(optim.init)(params)
    update_fn = jax.jit(optim.update)

    grads = jax.tree.map(jnp.ones_like, params)
    for _ in range(5):
      updates, state = update_fn(grads, state, params)
      np.testing.assert_almost_equal(state.hyperparams['b1'], 0.0)
      np.testing.assert_almost_equal(state.hyperparams['b2'], 0.0)
      np.testing.assert_almost_equal(state.hyperparams['eps'], 1e-8)
      np.testing.assert_almost_equal(state.hyperparams['eps_root'], 0.0)
      assert 'eps' in state.hyperparams
      test_utils.assert_trees_all_close(updates, grads)

  def test_overriding_hyperparam(self):
    optim = schedules.inject_hyperparams(transforms.clip_by_global_norm)(0.1)
    params = jnp.zeros((3, 5, 7))
    state = jax.jit(optim.init)(params)
    update_fn = jax.jit(optim.update)

    grads = jnp.ones_like(params)
    for i in range(5):
      state.hyperparams['max_norm'] = i
      updates, state = update_fn(grads, state)
      assert np.isclose(jnp.linalg.norm(updates.ravel()), i)

  @parameterized.named_parameters(('string', 'mask'), ('list', ['mask']))
  def test_static_args(self, static_args):
    @functools.partial(schedules.inject_hyperparams, static_args=static_args)
    def custom_optim(learning_rate, mask):
      return wrappers.masked(transform.scale(-learning_rate), mask)

    optim = custom_optim(
        0.1, functools.partial(jax.tree.map, lambda x: x.ndim > 1)
    )
    params = [jnp.ones((1, 2)), jnp.ones(2), jnp.ones((1, 1, 1))]
    grads = params
    state = jax.jit(optim.init)(params)
    updates, state = jax.jit(optim.update)(grads, state)
    expected_updates = jax.tree.map(
        lambda x: -0.1 * x if x.ndim > 1 else x, grads
    )

    assert set(state.hyperparams.keys()) == {'learning_rate'}, state.hyperparams
    test_utils.assert_trees_all_close(updates, expected_updates)

  @parameterized.named_parameters(('one_arg', 'b1'), ('two_arg', ['b1', 'b2']))
  def test_numeric_static_args(self, static_args):
    optim = schedules.inject_hyperparams(
        transform.scale_by_adam, static_args=static_args
    )(b1=0.9, b2=0.95)

    params = [jnp.ones((1, 2)), jnp.ones(2), jnp.ones((1, 1, 1))]
    grads = params
    state = jax.jit(optim.init)(params)
    _, state = jax.jit(optim.update)(grads, state)

    assert not set(state.hyperparams.keys()).intersection(set(static_args))

  def test_prng_key_not_hyperparameter(self):
    """Check that random.key can be handled by :func:``inject_hyperparams``."""

    def random_noise_optimizer(
        key: base.PRNGKey, scale: jax.Array
    ) -> base.GradientTransformation:
      def init_fn(params_like: base.Params) -> tuple[base.PRNGKey,
                                                     Union[jax.Array, float]]:
        del params_like
        return (key, scale)

      def update_fn(
          updates: base.Updates,
          state: tuple[base.PRNGKey, jax.Array],
          params: None = None,
      ) -> tuple[base.Updates, tuple[base.PRNGKey, Union[jax.Array, float]]]:
        del params
        key, scale = state
        keyit = iter(random.split(key, len(jax.tree.leaves(updates)) + 1))
        new_updates = jax.tree.map(
            lambda x: scale * random.normal(next(keyit), x.shape), updates
        )
        new_key = next(keyit)
        return new_updates, (new_key, scale)

      return base.GradientTransformation(init_fn, update_fn)

    optim = schedules.inject_hyperparams(random_noise_optimizer)(
        key=random.key(17), scale=1e-3
    )

    params = [jnp.ones((1, 2)), jnp.ones(2), jnp.ones((1, 1, 1))]
    grads = params
    state = jax.jit(optim.init)(params)
    _, state = jax.jit(optim.update)(grads, state)
    del state

  @parameterized.named_parameters(
      ('bf16hyp f32param bf16grad', jnp.bfloat16, jnp.float32, jnp.bfloat16),
      ('bf16hyp f32param f32_grads', jnp.bfloat16, jnp.float32, jnp.float32),
      ('f32hyp bf16param bf16grad', jnp.float32, jnp.bfloat16, jnp.bfloat16),
      ('f32hyp f32param bf16grad', jnp.float32, jnp.float32, jnp.bfloat16),
      ('f32hyp bf16param f32grad', jnp.float32, jnp.bfloat16, jnp.float32),
  )
  def test_hyperparam_dtypes(self, hyperparam_dtype, param_dtype, grad_dtype):
    """Tests that hyperparam dtype override works as desired."""
    optim = schedules.inject_hyperparams(
        transform.scale_by_adam, hyperparam_dtype=hyperparam_dtype
    )(b1=0.9, b2=0.95)

    params = [
        jnp.ones((1, 2), dtype=param_dtype),
        jnp.ones(2, dtype=param_dtype),
        jnp.ones((1, 1, 1), dtype=param_dtype),
    ]
    grads = jax.tree.map(lambda x: x.astype(grad_dtype), params)
    state = jax.jit(optim.init)(params)
    # Check that the hyperparams are overridden
    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

    _, state = jax.jit(optim.update)(grads, state)

    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

  @parameterized.named_parameters(('string', 'lr'), ('list', ['lr']))
  def test_static_args_error(self, static_args):
    with self.assertRaises(ValueError):
      schedules.inject_hyperparams(transform.scale, static_args=static_args)

  def test_inject_hyperparams_starts_with_step_count_zero(self):
    """Checks that inject_hyperparams uses step count 0 in the first update."""
    # See also: https://github.com/deepmind/optax/issues/415.
    opt = schedules.inject_hyperparams(transform.scale)(lambda count: count)
    params = jnp.zeros(3)
    grads = jnp.array([-1, 0, 1])
    updates, _ = jax.jit(opt.update)(grads, opt.init(params))
    np.testing.assert_array_equal(updates, np.zeros(3))


class StatefulTest(absltest.TestCase):

  def test_wrap_stateless_schedule(self):
    my_schedule = schedules.linear_schedule(1.0, 1.0, 10)
    my_wrapped_schedule = schedules.WrappedSchedule(my_schedule)

    count = jnp.zeros([], dtype=jnp.int32)
    state = my_wrapped_schedule.init()
    np.testing.assert_allclose(count, state, atol=0.0)

    for _ in range(8):
      np.testing.assert_allclose(
          my_schedule(count), my_wrapped_schedule(state), atol=0.0
      )
      count = count + 1
      extra_args = {'loss': jnp.ones([], dtype=jnp.float32)}
      state = my_wrapped_schedule.update(state, **extra_args)
      np.testing.assert_allclose(count, state, atol=0.0)

  def test_inject_stateful_hyperparams(self):
    grads = (
        jnp.ones((3,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
    )
    params = grads

    my_stateful_schedule = ExampleStatefulSchedule()
    tx = schedules.inject_hyperparams(transform.scale)(
        step_size=my_stateful_schedule
    )
    state = jax.jit(tx.init)(params)

    extra_args = {'addendum': 0.3 * jnp.ones((), dtype=jnp.float32)}
    _, state = jax.jit(tx.update)(
        grads, state, params=params, **extra_args
    )
    _, state = jax.jit(tx.update)(
        grads, state, params=params, **extra_args
    )

    lr = state.hyperparams['step_size']
    total = state.hyperparams_states['step_size']

    np.testing.assert_allclose(lr, extra_args['addendum'], atol=0.0)
    np.testing.assert_allclose(total, 2 * extra_args['addendum'], atol=0.0)


if __name__ == '__main__':
  absltest.main()
