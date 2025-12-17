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
"""Common tests for contributed optimizers.

Additional specific tests are implemented in additional files
"""

import functools
import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import test_utils
from optax._src import update
from optax._src import utils
from optax.schedules import _inject
from optax.schedules import _schedule
from optax.transforms import _accumulation
import optax.tree

# Testing contributions coded as GradientTransformations
_MAIN_OPTIMIZERS_UNDER_TEST = [
    {'opt_name': 'acprop', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'ademamix', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'simplified_ademamix', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'adopt', 'opt_kwargs': {'learning_rate': 1e-2}},
    {'opt_name': 'ano', 'opt_kwargs': {'learning_rate': 1e-3}},
    {'opt_name': 'cocob', 'opt_kwargs': {}},
    {'opt_name': 'cocob', 'opt_kwargs': {'weight_decay': 1e-2}},
    {'opt_name': 'dadapt_adamw', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'dog', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'dowg', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'momo', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'momo_adam', 'opt_kwargs': {'learning_rate': 1e-1}},
    {'opt_name': 'muon', 'opt_kwargs': {'learning_rate': 1e-2}},
    {'opt_name': 'prodigy', 'opt_kwargs': {'learning_rate': 1e-1}},
    {
        'opt_name': 'schedule_free_sgd',
        'opt_kwargs': {'learning_rate': 1e-2, 'warmup_steps': 5000},
    },
    {
        'opt_name': 'schedule_free_adamw',
        'opt_kwargs': {'learning_rate': 1e-2, 'warmup_steps': 5000},
    },
    {
        'opt_name': 'sophia',
        'opt_kwargs': {'learning_rate': 1e-2}
    },
]
for optimizer in _MAIN_OPTIMIZERS_UNDER_TEST:
  optimizer['wrapper_name'] = None
  optimizer['wrapper_kwargs'] = None

# Testing contributions coded as wrappers
# (just with sgd as we just want the behavior of the wrapper)
_MAIN_OPTIMIZERS_UNDER_TEST += [
    {
        'opt_name': 'sgd',
        'opt_kwargs': {'learning_rate': 1e-1},
        'wrapper_name': 'mechanize',
        'wrapper_kwargs': {'weight_decay': 0.0},
    },
    {
        'opt_name': 'sgd',
        'opt_kwargs': {'learning_rate': 1e-2},
        'wrapper_name': 'schedule_free',
        'wrapper_kwargs': {'learning_rate': 1e-2},
    },
    {
        'opt_name': 'sgd',
        'opt_kwargs': {'learning_rate': 1e-3},
        'wrapper_name': 'reduce_on_plateau',
        'wrapper_kwargs': {},
    },
]

# Adding here instantiations of wrappers with any base optimizer
_BASE_OPTIMIZERS = [
    {'opt_name': 'sgd', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'sgd', 'opt_kwargs': {'learning_rate': 1.0, 'momentum': 0.9}},
    {'opt_name': 'adam', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'adamw', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'adamax', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'adamaxw', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'adan', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'amsgrad', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'lamb', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'lion', 'opt_kwargs': {'learning_rate': 1.0, 'b1': 0.99}},
    {
        'opt_name': 'noisy_sgd',
        'opt_kwargs': {'learning_rate': 1.0, 'eta': 1e-4, 'key': 0},
    },
    {'opt_name': 'novograd', 'opt_kwargs': {'learning_rate': 1.0}},
    {
        'opt_name': 'optimistic_gradient_descent',
        'opt_kwargs': {'learning_rate': 1.0, 'alpha': 0.7, 'beta': 0.1},
    },
    {'opt_name': 'rmsprop', 'opt_kwargs': {'learning_rate': 1.0}},
    {
        'opt_name': 'rmsprop',
        'opt_kwargs': {'learning_rate': 1.0, 'momentum': 0.9},
    },
    {'opt_name': 'adabelief', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'radam', 'opt_kwargs': {'learning_rate': 1.0}},
    {'opt_name': 'sm3', 'opt_kwargs': {'learning_rate': 3.0}},
    {'opt_name': 'yogi', 'opt_kwargs': {'learning_rate': 1.0, 'b1': 0.99}},
]
# TODO(harshm): make LARS and Fromage work with mechanic.
_OTHER_OPTIMIZERS_UNDER_TEST = [
    {
        'opt_name': base_opt['opt_name'],
        'opt_kwargs': base_opt['opt_kwargs'],
        'wrapper_name': 'mechanize',
        'wrapper_kwargs': {'weight_decay': 0.0},
    }
    for base_opt in _BASE_OPTIMIZERS
]

_ALL_OPTIMIZERS_UNDER_TEST = tuple(
    _MAIN_OPTIMIZERS_UNDER_TEST + _OTHER_OPTIMIZERS_UNDER_TEST
)
_MAIN_OPTIMIZERS_UNDER_TEST = tuple(_MAIN_OPTIMIZERS_UNDER_TEST)


def _get_opt_factory(opt_name):
  """Get optimizer factory."""
  if hasattr(contrib, opt_name):
    return getattr(contrib, opt_name)
  if hasattr(alias, opt_name):
    return getattr(alias, opt_name)
  raise ValueError(f'Unknown optimizer: {opt_name}')


def _wrap_opt(opt, wrapper_name, wrapper_kwargs):
  if wrapper_name == 'reduce_on_plateau':
    return combine.chain(opt, contrib.reduce_on_plateau(**wrapper_kwargs))
  return getattr(contrib, wrapper_name)(opt, **wrapper_kwargs)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  def obj_fn(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, obj_fn


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  def obj_fn(params):
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  return initial_params, final_params, obj_fn


def _setup_matrix_parabola(dtype):
  """Quadratic function as an optimization target with 2D tensor parameters."""
  initial_params = jnp.zeros((2, 2), dtype=dtype)
  final_params = jnp.array([[3.0, -2.0], [1.0, 4.0]], dtype=dtype)

  def obj_fn(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, obj_fn


def _setup_mixed_tensor_target(dtype):
  """Optimization target combining 1D and 2D tensor parameters."""
  initial_1d_params = jnp.zeros((3,), dtype=dtype)
  final_1d_params = jnp.array([1.0, -1.0, 2.0], dtype=dtype)

  initial_2d_params = jnp.zeros((2, 2), dtype=dtype)
  final_2d_params = jnp.array([[1.0, 0.0], [-1.0, 1.0]], dtype=dtype)

  def obj_fn(params):
    """Objective function combining 1D and 2D parameters."""
    params_1d, params_2d = params
    loss_1d = jnp.sum(numerics.abs_sq(params_1d - final_1d_params))
    loss_2d = jnp.sum(numerics.abs_sq(params_2d - final_2d_params))
    return loss_1d + loss_2d

  initial_params = (initial_1d_params, initial_2d_params)
  final_params = (final_1d_params, final_2d_params)

  return initial_params, final_params, obj_fn


class ContribTest(parameterized.TestCase):

  @parameterized.product(_ALL_OPTIMIZERS_UNDER_TEST, wrap=[True, False])
  def test_optimizers_accept_extra_args(
      self, opt_name, opt_kwargs, wrapper_name, wrapper_kwargs, wrap):
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrap and wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    # intentionally ommit: opt = base.with_extra_args_support(opt)

    initial_params, _, objective = _setup_rosenbrock(jnp.float32)

    @jax.jit
    def step(params, state):
      value, updates = jax.value_and_grad(objective)(params)
      update_kwargs = {'unexpected_extra_args_your_optimizer_doesnt_expect': 1}
      if opt_name in ['momo', 'momo_adam', 'sgd']:
        update_kwargs['value'] = value
      if opt_name in ['sophia']:
        update_kwargs['obj_fn'] = objective
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)

    with self.subTest('Test that update works with extra args'):
      for _ in range(2):
        params, state = step(params, state)

    with self.subTest('Test that the optimizer doesn\'t recompile on 2nd call'):
      params = initial_params
      state = opt.init(params)
      params, state = step(params, state)
      with test_utils.log_compilations() as compilation_logs:
        _ = step(params, state)
      self.assertEmpty(
          compilation_logs, 'Optimizer recompiles on second call to "update".'
      )

  @parameterized.product(
      _ALL_OPTIMIZERS_UNDER_TEST,
      target=(
          _setup_parabola,
          _setup_rosenbrock,
          _setup_matrix_parabola,
          _setup_mixed_tensor_target,
      ),
      dtype=('float32',),
  )
  def test_optimizers(
      self,
      opt_name,
      opt_kwargs,
      wrapper_name,
      wrapper_kwargs,
      target,
      dtype,
  ):
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    initial_params, final_params, obj_fn = target(dtype)

    @jax.jit
    def step(params, state):
      value, updates = jax.value_and_grad(obj_fn)(params)
      if (
          opt_name in ['momo', 'momo_adam']
          or wrapper_name == 'reduce_on_plateau'
      ):
        update_kwargs = {'value': value}
      elif opt_name == 'sophia':
        update_kwargs = {'obj_fn': obj_fn}
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

      def f(params_state, _):
        return step(*params_state), None

      (params, state), _ = jax.lax.scan(f, (params, state), length=30_000)

      if (
          opt_name in ['schedule_free_sgd', 'schedule_free_adamw']
          or wrapper_name == 'schedule_free'
      ):
        params = contrib.schedule_free_eval_params(state, params)
      test_utils.assert_trees_all_close(
          params, final_params, rtol=3e-2, atol=3e-2)

  @parameterized.product(_MAIN_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs, wrapper_name=None, wrapper_kwargs=None
  ):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/deepmind/optax/issues/412.
    # When debugging this, make sure that options like weight decay or not
    # are checked by asserting wehter such a value is None or not (see e.g. the
    # logic in schedule_free_adamw). Some hyperparameters may not be supported
    # by inject_hyperparams (e.g. warmup_steps). In that case (if you're sure
    # you can ignore such hyperparameter), add the exception below.
    if wrapper_name == 'reduce_on_plateau':
      # TODO(vroulet): discuss adding support for reduce_on_plateau
      # so removing all assertions in its definition
      self.skipTest('reduce_on_plateau is not supported by inject_hyperparams.')
    if wrapper_name is None:
      factory = _get_opt_factory(opt_name)
      hparams = opt_kwargs
    else:
      base_opt = _get_opt_factory(opt_name)(**opt_kwargs)
      factory = getattr(contrib, wrapper_name)
      factory = functools.partial(factory, base_opt)
      hparams = wrapper_kwargs
    opt = factory(**hparams)

    # Add here the hyperparameters that cannot be injected with
    # inject_hyperparams.
    static_args = []
    for uninjectable_hparam in ['warmup_steps', 'num_betas', 'clip_value_fn',
                                'ns_steps']:
      if uninjectable_hparam in inspect.signature(factory).parameters.keys():
        static_args.append(uninjectable_hparam)
    static_args = tuple(static_args)
    opt_inject = _inject.inject_hyperparams(factory, static_args)(**hparams)

    params = [jnp.negative(jnp.ones((2, 3))), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), jnp.negative(jnp.ones((2, 5, 2)))]

    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': jnp.array(1.0)}
    else:
      update_kwargs = {}
    if opt_name == 'sophia':
      obj_fn = lambda x: optax.tree.norm(x, squared=True)
      update_fn = functools.partial(opt.update, obj_fn=obj_fn)
      inject_update_fn = functools.partial(opt_inject.update, obj_fn=obj_fn)
    else:
      update_fn = opt.update
      inject_update_fn = opt_inject.update

    state = jax.jit(opt.init)(params)
    updates, new_state = jax.jit(update_fn)(
        grads, state, params, **update_kwargs
    )

    state_inject = jax.jit(opt_inject.init)(params)
    updates_inject, new_state_inject = jax.jit(inject_update_fn)(
        grads, state_inject, params, **update_kwargs
    )

    with self.subTest('Equality of updates.'):
      test_utils.assert_trees_all_close(updates_inject, updates, rtol=1e-5)
    with self.subTest('Equality of new optimizer states.'):
      test_utils.assert_trees_all_close(
          new_state_inject.inner_state, new_state, rtol=1e-5, atol=1e-5
      )

  @parameterized.product(
      _MAIN_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32')
  )
  def test_preserve_dtype(
      self, opt_name, opt_kwargs, dtype, wrapper_name=None, wrapper_kwargs=None
  ):
    """Test that the optimizers return updates of same dtype as params."""
    # When debugging this test, note that operations like
    # x = 0.5**jnp.asarray(1, dtype=jnp.int32)
    # (appearing in e.g. optax.tree.bias_correction)
    # are promoted (strictly) to float32 when jitted
    # see https://github.com/jax-ml/jax/issues/23337
    # This may end up letting updates have a dtype different from params.
    # The solution is to fix the dtype of the result to the desired dtype
    # (just as done in optax.tree.bias_correction).
    # Otherwise, just make sure that all variables defined in the optimizer have
    # the same dtype as the parameters.
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    value, grads = jax.value_and_grad(fun)(params)
    state = jax.jit(opt.init)(params)
    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': value}
    else:
      update_kwargs = {}
    if opt_name == 'sophia':
      update_fn = functools.partial(opt.update, obj_fn=fun)
    else:
      update_fn = opt.update
    updates, _ = jax.jit(update_fn)(grads, state, params, **update_kwargs)
    self.assertEqual(updates.dtype, params.dtype)

  @parameterized.product(
      _MAIN_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32')
  )
  def test_gradient_accumulation(
      self, opt_name, opt_kwargs, dtype, wrapper_name=None, wrapper_kwargs=None
  ):
    """Test that the optimizers can safely be used with optax.MultiSteps."""
    # Checks for issues like https://github.com/google-deepmind/optax/issues/377
    # Should pass as long as test_preserve_dtype passes.
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)

    fun = lambda x: jnp.sum(x**2)

    if opt_name == 'sophia':
      update_fn = functools.partial(opt.update, obj_fn=fun)
    else:
      update_fn = opt.update
    opt = base.GradientTransformationExtraArgs(opt.init, update_fn)
    opt = _accumulation.MultiSteps(opt, every_k_schedule=4)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    value, grads = jax.value_and_grad(fun)(params)
    state = jax.jit(opt.init)(params)
    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': value}
    else:
      update_kwargs = {}
    updates, _ = jax.jit(opt.update)(grads, state, params, **update_kwargs)
    test_utils.assert_trees_all_equal(updates, jnp.zeros_like(grads))

  @parameterized.product(
      _ALL_OPTIMIZERS_UNDER_TEST,
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
  )
  def test_state_shape_dtype_shard_stability(
      self, opt_name, opt_kwargs, wrapper_name, wrapper_kwargs, dtype):
    if dtype == jnp.complex128 and jax.default_backend() == 'tpu':
      self.skipTest('TPU backend does not support complex128')
    del wrapper_name, wrapper_kwargs  # Unused.
    with utils.x64_precision(dtype in (jnp.float64, jnp.complex128)):
      opt = _get_opt_factory(opt_name)(**opt_kwargs)
      initial_params, _, objective = _setup_parabola(dtype)

      @jax.jit
      def step(params, state):
        value, updates = jax.value_and_grad(objective)(params)
        value = value.astype(jnp.float16 if dtype != jnp.float16
                             else jnp.bfloat16)  # confuse dtype intentionally
        if opt_name in ['polyak_sgd', 'momo', 'momo_adam']:
          update_kwargs = {'value': value}
        elif opt_name == 'sophia':
          update_kwargs = {'obj_fn': objective}
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

  @parameterized.product(_ALL_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_accept_learning_rate_schedule_if_type_annotated_as_such(
      self, opt_name, opt_kwargs, wrapper_name, wrapper_kwargs):
    del wrapper_kwargs
    initial_params, _, obj_fn = _setup_parabola(jnp.float32)

    opt_setup = _get_opt_factory(opt_name)

    # check if the optimizer is annotated to accept a learning rate schedule
    opt_setup_signature = inspect.signature(opt_setup)
    lr_arg = opt_setup_signature.parameters.get('learning_rate', None)
    if lr_arg is None or lr_arg.annotation not in (base.ScalarOrSchedule,
                                                   base.Schedule):
      self.skipTest('Optimizer doesn\'t accept a learning schedule.')
    opt_kwargs_with_schedule = dict(
        opt_kwargs, learning_rate=_schedule.constant_schedule(1e-3))
    opt_kwargs_with_lr = dict(opt_kwargs, learning_rate=1e-3)

    opt_with_schedule = opt_setup(**opt_kwargs_with_schedule)
    opt_with_lr = opt_setup(**opt_kwargs_with_lr)

    def step(opt, params, state):
      value, updates = jax.value_and_grad(obj_fn)(params)
      if (
          opt_name in ['momo', 'momo_adam']
          or wrapper_name == 'reduce_on_plateau'
      ):
        update_kwargs = {'value': value}
      elif opt_name == 'sophia':
        update_kwargs = {'obj_fn': obj_fn}
      else:
        update_kwargs = {}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params_with_schedule, params_with_lr = initial_params, initial_params
    state_with_schedule = opt_with_schedule.init(params_with_schedule)
    state_with_lr = opt_with_lr.init(params_with_lr)
    with self.subTest('Test that optimization works'):
      for _ in range(10):
        params_with_schedule, state_with_schedule = step(
            opt_with_schedule, params_with_schedule, state_with_schedule)
        params_with_lr, state_with_lr = step(
            opt_with_lr, params_with_lr, state_with_lr)
      test_utils.assert_trees_all_close(params_with_schedule, params_with_lr)


if __name__ == '__main__':
  absltest.main()
