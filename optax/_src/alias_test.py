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
"""Tests for methods defined in `alias.py`."""


import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import numerics
from optax._src import test_utils
from optax._src import update
from optax._src import utils
from optax.schedules import _inject
from optax.transforms import _accumulation
import optax.tree


_OPTIMIZERS_UNDER_TEST = (
    {'opt_name': 'adabelief', 'opt_kwargs': {'learning_rate': 1e-2}},
    {'opt_name': 'adadelta', 'opt_kwargs': {'learning_rate': 0.1}},
    {'opt_name': 'adadelta', 'opt_kwargs': {}},
    {'opt_name': 'adafactor', 'opt_kwargs': {'learning_rate': 5e-3}},
    {'opt_name': 'adafactor', 'opt_kwargs': {}},
    {'opt_name': 'adagrad', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'adam', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'adamax', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'adamaxw', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'adamw', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'adan', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'amsgrad', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'fromage', 'opt_kwargs': {'learning_rate': 5e-3}},
    {'opt_name': 'lamb', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'lars', 'opt_kwargs': {'learning_rate': 1.0}},
    {
        'opt_name': 'lion',
        'opt_kwargs': {'learning_rate': 1e-2, 'weight_decay': 1e-4},
    },
    {'opt_name': 'nadam', 'opt_kwargs': {'learning_rate': 1e-2}},
    {'opt_name': 'nadamw', 'opt_kwargs': {'learning_rate': 1e-2}},
    {
        'opt_name': 'noisy_sgd',
        'opt_kwargs': {'learning_rate': 1e-3, 'eta': 1e-4, 'key': 0},
    },
    {'opt_name': 'novograd', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'optimistic_adam', 'opt_kwargs': {'learning_rate': 2e-3}},
    {'opt_name': 'optimistic_adam_v2', 'opt_kwargs': {'learning_rate': 2e-3}},
    {
        'opt_name': 'optimistic_gradient_descent',
        'opt_kwargs': {'learning_rate': 2e-3, 'alpha': 0.7, 'beta': 0.1},
    },
    {'opt_name': 'polyak_sgd', 'opt_kwargs': {'max_learning_rate': 1.0}},
    {'opt_name': 'radam', 'opt_kwargs': {'learning_rate': 5e-3}},
    {'opt_name': 'rmsprop', 'opt_kwargs': {'learning_rate': 5e-3}},
    {
        'opt_name': 'rmsprop',
        'opt_kwargs': {'learning_rate': 5e-3, 'momentum': 0.9},
    },
    {'opt_name': 'rprop', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'sgd', 'opt_kwargs': {'learning_rate': 1e-3, 'momentum': 0.9}},
    {'opt_name': 'sign_sgd', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'signum', 'opt_kwargs': {'learning_rate': 1e-2}},
    {'opt_name': 'sm3', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'yogi', 'opt_kwargs': {'learning_rate': 1e-1}},
)


def _get_opt(self: absltest.TestCase, opt_name: str):
  if opt_name == 'optimistic_adam':
    opt_ = getattr(alias, opt_name)

    @functools.wraps(opt_)
    def opt(*args, **kwargs):
      with self.assertWarnsRegex(
          DeprecationWarning, 'use `optimistic_adam_v2` instead'
      ):
        return opt_(*args, **kwargs)

    return opt

  return getattr(alias, opt_name)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  if jnp.iscomplexobj(dtype):
    final_params *= 1 + 1j

  def objective(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, objective


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  if jnp.iscomplexobj(dtype):
    a *= 1 + 1j

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  def objective(params):
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  return initial_params, final_params, objective


class AliasTest(parameterized.TestCase):

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32, jnp.complex64),
  )
  def test_optimization(self, opt_name, opt_kwargs, target, dtype):
    if opt_name in (
        'fromage',
        'noisy_sgd',
        'sm3',
        'optimistic_gradient_descent',
        'optimistic_adam',
        'lion',
        'rprop',
        'adadelta',
        'adan',
        'polyak_sgd',
        'sign_sgd',
        'signum',
        'lars',
    ) and jnp.iscomplexobj(dtype):
      raise absltest.SkipTest(
          f'{opt_name} does not support complex parameters.'
      )

    if opt_name in ('sign_sgd', 'signum') and target is _setup_rosenbrock:
      raise absltest.SkipTest(
          f'{opt_name} requires learning rate scheduling to solve the'
          ' Rosenbrockfunction'
      )

    opt = _get_opt(self, opt_name)(**opt_kwargs)
    initial_params, final_params, objective = target(dtype)

    @jax.jit
    def step(params, state):
      value, updates = jax.value_and_grad(objective)(params)
      # Complex gradients need to be conjugated before being added to parameters
      # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
      updates = jax.tree.map(lambda x: x.conj(), updates)
      if opt_name == 'polyak_sgd':
        update_kwargs = {'value': value}
      else:
        update_kwargs = {}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)

    with self.subTest('Test that tree_map_params works'):
      # A no-op change, to verify that tree map works.
      state = optax.tree.map_params(opt, lambda v: v, state)

    with self.subTest('Test that optimization works'):
      for it in range(10000):
        if it == 1:
          with test_utils.log_compilations() as compilation_logs:
            params, state = step(params, state)
          self.assertEmpty(compilation_logs,
                           'Optimizer recompiles on second call to "update".')
        else:
          params, state = step(params, state)

      if (opt_name in ('adadelta', 'adafactor')
          and opt_kwargs.get('learning_rate') is None):
        raise absltest.SkipTest(
            f'{opt_name} needs a non-None learning rate for numerically stable'
            ' optimization in practice.'
        )
      test_utils.assert_trees_all_close(
          params, final_params, rtol=3e-2, atol=3e-2)

    with self.subTest('Test that the optimizer doesn\'t recompile on 2nd call'):
      params = initial_params
      state = opt.init(params)
      params, state = step(params, state)
      with test_utils.log_compilations() as compilation_logs:
        _ = step(params, state)
      self.assertEmpty(
          compilation_logs, 'Optimizer recompiles on second call to "update".'
      )

  @parameterized.product(_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_accept_extra_args(self, opt_name, opt_kwargs):
    opt = _get_opt(self, opt_name)(**opt_kwargs)
    # intentionally ommit: opt = base.with_extra_args_support(opt)
    initial_params, _, objective = _setup_rosenbrock(jnp.float32)

    @jax.jit
    def step(params, state):
      value, updates = jax.value_and_grad(objective)(params)
      update_kwargs = {'unexpected_extra_args_your_optimizer_doesnt_expect': 1}
      if opt_name in ['polyak_sgd']:
        update_kwargs = {'value': value}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)

    with self.subTest('Test that update works with extra values'):
      for _ in range(2):
        params, state = step(params, state)

  @parameterized.product(_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs
  ):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/google-deepmind/optax/issues/412.
    opt_factory = _get_opt(self, opt_name)
    opt = opt_factory(**opt_kwargs)
    if opt_name == 'adafactor':
      # Adafactor wrapped in inject_hyperparams currently needs a static
      # argument to be specified in order to be jittable. See issue
      # https://github.com/google-deepmind/optax/issues/412.
      opt_inject = _inject.inject_hyperparams(
          opt_factory, static_args=('min_dim_size_to_factor',)
      )(**opt_kwargs)
    else:
      opt_inject = _inject.inject_hyperparams(opt_factory)(**opt_kwargs)

    params = [jnp.negative(jnp.ones((2, 3))), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), jnp.negative(jnp.ones((2, 5, 2)))]

    state = jax.jit(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': jnp.array(0.0)}
    else:
      update_kwargs = {}
    updates, new_state = jax.jit(opt.update)(
        grads, state, params, **update_kwargs
    )

    state_inject = jax.jit(opt_inject.init)(params)
    updates_inject, new_state_inject = jax.jit(opt_inject.update)(
        grads, state_inject, params, **update_kwargs
    )

    with self.subTest('Equality of updates.'):
      test_utils.assert_trees_all_close(updates_inject, updates, rtol=1e-3)
    with self.subTest('Equality of new optimizer states.'):
      test_utils.assert_trees_all_close(
          optax.tree.unwrap_random_key_data(new_state_inject.inner_state),
          optax.tree.unwrap_random_key_data(new_state),
          rtol=1e-4,
      )

  @parameterized.product(
      params_dtype=('bfloat16', 'float32', 'complex64', None),
      state_dtype=('bfloat16', 'float32', 'complex64', None),
      opt_name=('sgd_mom', 'adam', 'adamw'),
  )
  def test_explicit_dtype(self, params_dtype, state_dtype, opt_name):
    if opt_name == 'sgd_mom':
      opt = alias.sgd(0.1, momentum=0.9, accumulator_dtype=state_dtype)
      attribute_name = 'trace'
    elif opt_name in ['adam', 'adamw']:
      opt = _get_opt(self, opt_name)(0.1, mu_dtype=state_dtype)
      attribute_name = 'mu'
    else:
      raise ValueError(f'Unsupported optimizer: {opt_name}')

    params_dtype = jax.dtypes.canonicalize_dtype(params_dtype)
    params = jnp.array([0.0, 0.0], dtype=params_dtype)
    state = opt.init(params)

    with self.subTest('Test that attribute dtype is correct'):
      if state_dtype is None:
        expected_dtype = params_dtype
      else:
        expected_dtype = jax.dtypes.canonicalize_dtype(state_dtype)
      attribute = optax.tree.get(state, attribute_name)
      self.assertEqual(expected_dtype, attribute.dtype)

  @parameterized.product(_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32'))
  def test_preserve_dtype(self, opt_name, opt_kwargs, dtype):
    """Test that the optimizers return updates of same dtype as gradients."""
    # When debugging this test, note that operations like
    # x = 0.5**jnp.asarray(1, dtype=jnp.int32)
    # (appearing in e.g. optax.tree.bias_correction)
    # are promoted (strictly) to float32 when jitted
    # see https://github.com/jax-ml/jax/issues/23337
    # This may end up letting updates have a dtype different from params.
    # The solution is to fix the dtype of the result to the desired dtype
    # (just as done in optax.tree.bias_correction).
    dtype = jnp.dtype(dtype)
    opt_factory = _get_opt(self, opt_name)
    opt = opt_factory(**opt_kwargs)
    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = jax.jit(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': fun(params)}
    else:
      update_kwargs = {}
    updates, _ = jax.jit(opt.update)(grads, state, params, **update_kwargs)
    self.assertEqual(updates.dtype, grads.dtype)

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64, jnp.complex64,
             jnp.complex128),
  )
  def test_state_shape_dtype_shard_stability(self, opt_name, opt_kwargs, dtype):
    if dtype == jnp.complex128 and jax.default_backend() == 'tpu':
      self.skipTest('TPU backend does not support complex128')
    if opt_name in (
        'fromage', 'noisy_sgd', 'sm3', 'optimistic_gradient_descent',
        'optimistic_adam', 'lion', 'rprop', 'adadelta', 'adan', 'polyak_sgd',
        'sign_sgd', 'signum') and jnp.iscomplexobj(dtype):
      raise absltest.SkipTest(
          f'{opt_name} does not support complex parameters.'
      )

    with utils.x64_precision(dtype in (jnp.float64, jnp.complex128)):
      opt = _get_opt(self, opt_name)(**opt_kwargs)
      initial_params, _, objective = _setup_parabola(dtype)

      @jax.jit
      def step(params, state):
        value, updates = jax.value_and_grad(objective)(params)
        # Complex gradients need to be conjugated before being added to
        # parameters
        # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
        updates = jax.tree.map(lambda x: x.conj(), updates)
        value = value.astype(jnp.float16 if dtype != jnp.float16
                             else jnp.float32)
        if opt_name == 'polyak_sgd':
          update_kwargs = {'value': value}
        elif opt_name == 'lbfgs':
          update_kwargs = {'value': value, 'grad': updates,
                           'value_fn': objective}
        else:
          update_kwargs = {}
        # defeat compiler optimization, use lax.cond to check for stability
        cond = jax.random.randint(jax.random.key(0), (), 0, 2) == 0
        updates, state = jax.lax.cond(
            cond, lambda: opt.update(updates, state, params, **update_kwargs),
            lambda: (params, state))
        params = update.apply_updates(params, updates)
        return params, state

      params = initial_params
      state = opt.init(params)

      with self.subTest('Test that update is dtype stable'):
        for _ in range(2):
          params, new_state = step(params, state)
          assert jax.tree.leaves(params)[0].dtype == dtype
          cond = jax.random.randint(jax.random.key(0), (), 0, 2) == 0
          # pylint: disable=cell-var-from-loop
          state = jax.lax.cond(cond, lambda: state,  # noqa: B023
                               lambda: new_state)    # noqa: B023
          # pylint: enable=cell-var-from-loop

  @parameterized.product(_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32'))
  def test_gradient_accumulation(self, opt_name, opt_kwargs, dtype):
    """Test that the optimizers can safely be used with optax.MultiSteps."""
    # Checks for issues like https://github.com/google-deepmind/optax/issues/377
    dtype = jnp.dtype(dtype)
    opt_factory = _get_opt(self, opt_name)

    base_opt = opt_factory(**opt_kwargs)
    opt = _accumulation.MultiSteps(base_opt, every_k_schedule=4)

    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = jax.jit(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': fun(params)}
    elif opt_name == 'lbfgs':
      update_kwargs = {'value': fun(params), 'grad': grads, 'value_fn': fun}
    else:
      update_kwargs = {}

    static_kwargs = {
        k: v for k, v in update_kwargs.items() if not isinstance(v, jax.Array)}
    dyn_kwargs = {
        k: v for k, v in update_kwargs.items() if isinstance(v, jax.Array)}
    updates, _ = jax.jit(functools.partial(opt.update, **static_kwargs))(
        grads, state, params, **dyn_kwargs)
    test_utils.assert_trees_all_equal(updates, jnp.zeros_like(grads))


if __name__ == '__main__':
  absltest.main()
