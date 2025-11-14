# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods defined in `linesearch.py`."""

from collections.abc import Callable
import contextlib
import functools
import io
import itertools
import math
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jrd
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import linesearch as _linesearch
from optax._src import test_utils
from optax._src import transform
from optax._src import update
from optax._src import utils
import optax.tree
import scipy.optimize as scipy_optimize


def get_problem(name: str):
  """Objectives to test linesearches on."""

  def polynomial(x):
    return -x - x**3 + x**4

  def exponential(x):
    return jnp.exp(-4 * x) + x**2

  def sinusoidal(x):
    return -jnp.sin(10 * x)

  def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

  def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2

  def matyas(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

  def eggholder(x):
    return -(x[1] + 47) * jnp.sin(
        jnp.sqrt(jnp.abs(x[0] / 2.0 + x[1] + 47.0))
    ) - x[0] * jnp.sin(jnp.sqrt(jnp.abs(x[0] - (x[1] + 47.0))))

  def zakharov(x):
    ii = jnp.arange(1, len(x) + 1, step=1, dtype=x.dtype)
    sum1 = (x**2).sum()
    sum2 = (0.5 * ii * x).sum()
    return sum1 + sum2**2 + sum2**4

  problems = {
      'polynomial': {'fn': polynomial, 'input_shape': ()},
      'exponential': {'fn': exponential, 'input_shape': ()},
      'sinusoidal': {'fn': sinusoidal, 'input_shape': ()},
      'rosenbrock': {'fn': rosenbrock, 'input_shape': (16,)},
      'himmelblau': {'fn': himmelblau, 'input_shape': (2,)},
      'matyas': {'fn': matyas, 'input_shape': (2,)},
      'eggholder': {'fn': eggholder, 'input_shape': (2,)},
      'zakharov': {'fn': zakharov, 'input_shape': (6,)},
  }
  return problems[name]


class BacktrackingLinesearchTest(parameterized.TestCase):

  def _check_decrease_conditions(
      self, fun, init_params, descent_dir, final_params, final_state, opt_args
  ):
    """Check decrease conditions."""
    init_value, init_grad = jax.value_and_grad(fun)(init_params)
    final_value = fun(final_params)
    final_lr = final_state[0]

    slope = optax.tree.vdot(descent_dir, init_grad)
    slope_rtol, atol, rtol = (
        opt_args['slope_rtol'],
        opt_args['atol'],
        opt_args['rtol'],
    )
    sufficient_decrease = (
        final_value
        <= (1 + rtol) * init_value + slope_rtol * final_lr * slope + atol
    )
    self.assertTrue(sufficient_decrease)

  def test_linesearch_update(self):
    fun = lambda x: jnp.sum(x**2)
    params = jnp.zeros(2)
    updates = -jax.grad(fun)(params)

    opt = _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=5)

    state = opt.init(params)

    # At initialization the value is inf
    value = optax.tree.get(state, 'value')
    self.assertTrue(jnp.isinf(value))

    update_fn = functools.partial(opt.update, value_fn=fun)
    update_fn = jax.jit(update_fn)
    _, state = update_fn(
        updates, state, params, value=fun(params), grad=jax.grad(fun)(params)
    )

    # If the step worked the value in the state should not be inf after one step
    value = optax.tree.get(state, 'value')
    self.assertFalse(jnp.isinf(value))

  @parameterized.product(
      problem_name=['rosenbrock', 'himmelblau', 'matyas', 'eggholder'],
      increase_factor=[1.0, 1.5, math.inf],
      slope_rtol=[1e-4, 0.0],
      atol=[1e-4, 0.0],
      rtol=[1e-4, 0.0],
      seed=[0, 1],
  )
  def test_linesearch(
      self,
      problem_name,
      increase_factor,
      slope_rtol,
      atol,
      rtol,
      seed,
  ):
    """Test backtracking linesearch (single update step)."""
    key = jrd.key(seed)
    problem = get_problem(problem_name)
    fn, input_shape = problem['fn'], problem['input_shape']
    init_params = jrd.normal(key, input_shape)

    base_opt = alias.sgd(learning_rate=1.0)
    descent_dir = -jax.grad(fn)(init_params)

    opt_args = {
        'max_backtracking_steps': 50,
        'slope_rtol': slope_rtol,
        'increase_factor': increase_factor,
        'atol': atol,
        'rtol': rtol,
    }

    solver = combine.chain(
        base_opt,
        _linesearch.scale_by_backtracking_linesearch(**opt_args),
    )
    init_state = solver.init(init_params)

    value, grad = jax.value_and_grad(fn)(init_params)
    updates, state = solver.update(
        grad, init_state, init_params, value=value, grad=grad, value_fn=fn
    )
    params = update.apply_updates(init_params, updates)

    self._check_decrease_conditions(
        fn, init_params, descent_dir, params, state[-1], opt_args
    )

  def test_gradient_descent_with_linesearch(self):
    init_params = jnp.array([-1.0, 10.0, 1.0])
    final_params = jnp.array([1.0, -1.0, 1.0])

    def fn(params):
      return jnp.sum((params - final_params) ** 2)

    # Base optimizer with a large learning rate to see if the linesearch works.
    base_opt = alias.sgd(learning_rate=10.0)
    solver = combine.chain(
        base_opt,
        _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
    )
    init_state = solver.init(init_params)
    max_iter = 40

    update_fn = functools.partial(solver.update, value_fn=fn)
    update_fn = jax.jit(update_fn)
    params = init_params
    state = init_state
    for _ in range(max_iter):
      value, grad = jax.value_and_grad(fn)(params)
      updates, state = update_fn(grad, state, params, value=value, grad=grad)
      params = update.apply_updates(params, updates)
    test_utils.assert_trees_all_close(
        final_params, params, atol=1e-2, rtol=1e-2)

  def test_recycling_value_and_grad(self):
    # A vmap or a pmap makes the cond in value_and_state_from_grad
    # become a select and in that case this code cannot be optimal.
    # So we skip the pmap test.
    init_params = jnp.array([1.0, 10.0, 1.0])
    final_params = jnp.array([1.0, -1.0, 1.0])

    def fn(params):
      return jnp.sum((params - final_params) ** 2)

    value_and_grad = _linesearch.value_and_grad_from_state(fn)

    base_opt = alias.sgd(learning_rate=0.1)
    solver = combine.chain(
        base_opt,
        _linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=15,
            increase_factor=1.0,
            slope_rtol=0.5,
            store_grad=True,
        ),
    )
    init_state = solver.init(init_params)
    max_iter = 40

    update_fn = functools.partial(solver.update, value_fn=fn)

    def fake_fun(_):
      return 1.0

    fake_value_and_grad = _linesearch.value_and_grad_from_state(fake_fun)

    def step_(params, state, iter_num):
      # Should still work as the value and grad are extracted from the state
      value, grad = jax.lax.cond(
          iter_num > 0,
          lambda: fake_value_and_grad(params, state=state),
          lambda: value_and_grad(params, state=state),
      )
      updates, state = update_fn(grad, state, params, value=value, grad=grad)
      params = update.apply_updates(params, updates)
      return params, state

    step = jax.jit(step_)
    params = init_params
    state = init_state
    for iter_num in range(max_iter):
      params, state = step(params, state, iter_num)
    params = jax.block_until_ready(params)
    test_utils.assert_trees_all_close(
        final_params, params, atol=1e-2, rtol=1e-2)

  def test_armijo_sgd(self):
    def fn(params, x, y):
      return jnp.sum((x.dot(params) - y) ** 2)

    # Create artificial data
    noise = 1e-3
    key = jrd.key(0)
    x_key, y_key, params_key = jrd.split(key, 3)
    d, m, n = 2, 16, 2
    xs = jrd.normal(x_key, (n, m, d))
    target_params = jrd.normal(params_key, (d,))
    ys = jnp.stack([x.dot(target_params) for x in xs])
    ys = ys + noise * jrd.normal(y_key, (n, m))
    xs_iter = itertools.cycle(iter(xs))
    ys_iter = itertools.cycle(iter(ys))
    init_params = jnp.zeros((d,))

    base_opt = alias.sgd(learning_rate=1.0)

    solver = combine.chain(
        base_opt,
        _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
    )
    num_passes = 10

    state = solver.init(init_params)
    update_fn = functools.partial(solver.update, value_fn=fn)
    update_fn = jax.jit(update_fn)

    value_and_grad_fn = jax.jit(jax.value_and_grad(fn))

    params = init_params
    for _ in range(num_passes):
      x, y = next(xs_iter), next(ys_iter)
      value, grad = value_and_grad_fn(params, x, y)
      updates, state = update_fn(
          grad, state, params, value=value, grad=grad, x=x, y=y
      )
      params = update.apply_updates(params, updates)
    test_utils.assert_trees_all_close(
        params, target_params, atol=5 * 1e-2, rtol=5 * 1e-2
    )

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
      confuse_dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
  )
  def test_dtype_stability(self, dtype, confuse_dtype):
    kw = ['slope_rtol', 'decrease_factor', 'increase_factor',
          'max_learning_rate', 'atol', 'rtol']
    with utils.x64_precision(True):
      # pytype: disable=wrong-arg-types
      opt = _linesearch.scale_by_backtracking_linesearch(
          max_backtracking_steps=5,
          **{k: jnp.array(1e-5, dtype=confuse_dtype) for k in kw})
      # pytype: enable=wrong-arg-types
      x = jnp.array([1.0, 2.0], dtype=dtype)
      state = opt.init(x)
      value_fn = lambda x: jnp.sum(x**2).astype(confuse_dtype)
      # TODO(rdyro): ensure optimizer updates dtype matches parameters
      cond = jax.random.randint(jax.random.key(0), (), 0, 2) == 0
      jax.lax.cond(cond, lambda x: opt.update(x, state, x, value=1.0, grad=x,
                                              value_fn=value_fn)[1],
                   lambda x: state, x)


def _run_linesearch(
    opt: base.GradientTransformationExtraArgs,
    fn: Callable[..., jax.typing.ArrayLike],
    params: base.Params,
    updates: base.Updates,
    stepsize_guess: Optional[jax.typing.ArrayLike] = None,
) -> tuple[base.Params, base.OptState]:
  """Runs the linesearch, i.e., a single update of scale_by_zoom_linesearch."""
  init_state = opt.init(params)
  if stepsize_guess is not None:
    init_state = optax.tree.set(init_state, learning_rate=stepsize_guess)

  value, grad = jax.value_and_grad(fn)(params)
  updates, final_state = opt.update(
      updates,
      init_state,
      params,
      value=value,
      grad=grad,
      value_fn=fn,
  )
  final_params = update.apply_updates(params, updates)
  return final_params, final_state


class ZoomLinesearchTest(parameterized.TestCase):

  def _check_value_and_grad_in_zoom_state(
      self,
      final_params: base.Params,
      final_state: base.OptState,
      value_fn: Callable[..., jax.typing.ArrayLike],
  ) -> None:
    # Check that the value and gradient stored in the state
    # match the value and gradient of the step done with the stepsize found
    final_value = optax.tree.get(final_state, 'value')
    final_grad = optax.tree.get(final_state, 'grad')
    test_utils.assert_trees_all_close(
        value_fn(final_params), final_value, atol=1e-5, rtol=1e-5
    )
    test_utils.assert_trees_all_close(
        jax.grad(value_fn)(final_params), final_grad, atol=1e-5, rtol=1e-5
    )

  def _check_linesearch_conditions(
      self,
      fun: Callable[..., jax.typing.ArrayLike],
      init_params: base.Params,
      updates: base.Updates,
      final_params: base.Params,
      final_state: base.OptState,
      opt_args: dict[str, jax.typing.ArrayLike],
      allow_failure: bool = False,
  ):
    """Check decrease conditions."""
    value_init, grad_init = jax.value_and_grad(fun)(init_params)
    value_final, grad_final = jax.value_and_grad(fun)(final_params)
    final_lr = optax.tree.get(final_state, 'learning_rate')
    num_linesearch_steps = optax.tree.get(final_state, 'num_linesearch_steps')
    if allow_failure:
      potentially_failed = (
          num_linesearch_steps > opt_args['max_linesearch_steps']
      )
    else:
      potentially_failed = False
    slope_init = optax.tree.vdot(updates, grad_init)
    slope_final = optax.tree.vdot(updates, grad_final)
    default_opt_args = {
        'slope_rtol': 1e-4,
        'curv_rtol': 0.9,
        'tol': 0.0,
    }
    opt_args = default_opt_args | opt_args
    slope_rtol, curv_rtol, tol = (
        opt_args['slope_rtol'],
        opt_args['curv_rtol'],
        opt_args['tol'],
    )
    with self.subTest('Check decrease conditions'):
      sufficient_decrease_error = value_final - (
          value_init + slope_rtol * final_lr * slope_init + tol
      )
      self.assertTrue(
          (sufficient_decrease_error <= 0) or potentially_failed,
          f'Sufficent decrease error: {sufficient_decrease_error}',
      )
    with self.subTest('Check curvature conditions'):
      small_curvature_error = jnp.abs(slope_final) - (
          curv_rtol * jnp.abs(slope_init) + tol
      )

      self.assertTrue(
          (small_curvature_error <= 0) or potentially_failed,
          f'Small curvature error: {small_curvature_error}',
      )

  def test_linesearch_update(self):
    fun = lambda x: jnp.sum(x**2)
    params = jnp.zeros(2)
    updates = -jax.grad(fun)(params)

    opt = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=5)

    state = opt.init(params)

    # At initialization the value is inf
    value = optax.tree.get(state, 'value')
    self.assertTrue(jnp.isinf(value))

    update_fn = functools.partial(opt.update, value_fn=fun)
    update_fn = jax.jit(update_fn)
    _, state = update_fn(
        updates, state, params, value=fun(params), grad=jax.grad(fun)(params)
    )

    # If the step worked the value in the state should not be inf after one step
    value = optax.tree.get(state, 'value')
    self.assertFalse(jnp.isinf(value))

  @absltest.skip('TODO(rdyro): need to match scipy linesearch algorithm')
  @parameterized.product(
      problem_name=[
          'polynomial',
          'exponential',
          'sinusoidal',
          'rosenbrock',
          'himmelblau',
          'matyas',
          'eggholder',
      ],
      seed=[0, 1],
  )
  def test_linesearch(self, problem_name: str, seed: int):
    """Test backtracking linesearch (single update step)."""
    # Fixed tolerances, we check the behavior in standard conditions
    slope_rtol = 1e-4
    curv_rtol = 0.9
    tol = 0.0

    key = jrd.key(seed)
    params_key, precond_key = jrd.split(key, 2)
    problem = get_problem(problem_name)
    fn, input_shape = problem['fn'], problem['input_shape']

    init_params = jrd.normal(params_key, input_shape)
    precond_vec = jrd.uniform(precond_key, input_shape)

    # Mimics a preconditioning by a diagonal matrix with non-negative entries
    # (non-negativity ensures that we keep a descent direction)
    init_updates = -precond_vec * jax.grad(fn)(init_params)

    opt_args = {
        'max_linesearch_steps': 30,
        'slope_rtol': slope_rtol,
        'curv_rtol': curv_rtol,
        'tol': tol,
        'max_learning_rate': None,
    }

    opt = _linesearch.scale_by_zoom_linesearch(**opt_args)
    final_params, final_state = _run_linesearch(
        opt, fn, init_params, init_updates
    )

    scipy_res = scipy_optimize.line_search(
        fn, jax.grad(fn), init_params, init_updates
    )
    with self.subTest('Check value and grad in zoom state'):
      self._check_value_and_grad_in_zoom_state(
          final_params, final_state, value_fn=fn
      )
    with self.subTest('Check linesearch conditions'):
      self._check_linesearch_conditions(
          fn, init_params, init_updates, final_params, final_state, opt_args
      )
    with self.subTest('Check against scipy'):
      stepsize = optax.tree.get(final_state, 'learning_rate')
      final_value = optax.tree.get(final_state, 'value')
      test_utils.assert_trees_all_close(scipy_res[0], stepsize, rtol=1e-5)
      test_utils.assert_trees_all_close(scipy_res[3], final_value, rtol=1e-5)

  def test_failure_descent_direction(self):
    """Check failure when updates are not a descent direction."""
    # jax.debug.print appears incompatible for testing on tpu
    # so we skip the test (and we cannot just use
    # @absltest.skipIf(jax.devices()[0] == 'tpu', reason = ...))
    # because jax arrays cannot be manipulated from the top level of the python
    # program
    if jax.default_backend() in ['tpu', 'gpu']:
      return

    # For this f and p, starting at a point on axis 0, the strong Wolfe
    # condition 2 is met if and only if the step length s satisfies
    # |x + s| <= c2 * |x|
    def fn(w):
      return jnp.dot(w, w)

    u = jnp.array([1.0, 0.0])
    w = 60 * u

    # Test that the line search fails for p not a descent direction
    # For high maxiter, still finds a decrease error because of
    # the approximate Wolfe condition so we reduced maxiter
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=18, curv_rtol=0.5, verbose=True
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      _, state = _run_linesearch(opt, fn, w, u, stepsize_guess=1.0)
      stepsize = optax.tree.get(state, 'learning_rate')
    # Check that we were not able to make a step or an infinitesimal one
    self.assertLess(stepsize, 1e-5)
    self.assertIn(_linesearch.FLAG_NOT_A_DESCENT_DIRECTION, stdout.getvalue())
    self.assertIn(_linesearch.FLAG_NO_STEPSIZE_FOUND, stdout.getvalue())

  def test_failure_too_small_max_stepsize(self):
    """Check failure when the max stepsize is too small."""
    # jax.debug.print appears incompatible for testing on tpu
    # so we skip the test (and we cannot just use
    # @absltest.skipIf(jax.devices()[0] == 'tpu', reason = ...))
    # because jax arrays cannot be manipulated from the top level of the python
    # program
    if jax.default_backend() in ['tpu', 'gpu']:
      return

    def fn(x):
      return jnp.dot(x, x)

    u = jnp.array([1.0, 0.0])
    w = -60 * u

    # Test that the line search fails if the maximum stepsize is too small
    # Here, smallest s satisfying strong Wolfe conditions for c2=0.5 is 30
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=10,
        curv_rtol=0.5,
        verbose=True,
        max_learning_rate=10.0,
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      _, state = _run_linesearch(opt, fn, w, u, stepsize_guess=1.0)
    stepsize = optax.tree.get(state, 'learning_rate')
    # Check that we still made a step
    self.assertEqual(stepsize, 10.0)
    self.assertIn(_linesearch.FLAG_INTERVAL_NOT_FOUND, stdout.getvalue())
    self.assertIn(
        _linesearch.FLAG_CURVATURE_COND_NOT_SATISFIED, stdout.getvalue()
    )

  def test_failure_not_enough_iter(self):
    """Check failure for a very small number of iterations."""
    # jax.debug.print appears incompatible for testing on tpu
    # so we skip the test (and we cannot just use
    # @absltest.skipIf(jax.devices()[0] == 'tpu', reason = ...))
    # because jax arrays cannot be manipulated from the top level of the python
    # program
    if jax.default_backend() in ['tpu', 'gpu']:
      return

    def fn(x):
      return jnp.dot(x, x)

    u = jnp.array([1.0, 0.0])
    w = -60 * u

    curv_rtol = 0.5
    # s=30 will only be tried on the 6th iteration, so this fails because
    # the maximum number of iterations is reached.
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=5, curv_rtol=curv_rtol, verbose=True
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      _, final_state = _run_linesearch(opt, fn, w, u, stepsize_guess=1.0)
    stepsize = optax.tree.get(final_state, 'learning_rate')
    # Check that we still made a step
    self.assertEqual(stepsize, 16.0)
    decrease_error = optax.tree.get(final_state, 'decrease_error')
    curvature_error = optax.tree.get(final_state, 'curvature_error')
    success = (decrease_error <= 0.0) and (curvature_error <= 0.0)
    self.assertFalse(success)
    # Here the error should not be that we haven't had a descent direction
    self.assertNotIn(
        _linesearch.FLAG_NOT_A_DESCENT_DIRECTION, stdout.getvalue()
    )

    # Check if it works normally
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=30, curv_rtol=curv_rtol
    )
    final_params, final_state = _run_linesearch(
        opt, fn, w, u, stepsize_guess=1.0
    )
    s = optax.tree.get(final_state, 'learning_rate')
    self._check_linesearch_conditions(
        fn, w, u, final_params, final_state, {'curv_rtol': curv_rtol}
    )
    self.assertGreaterEqual(s, 30.0)

  def test_failure_flat_fun(self):
    """Check failure for a very flat function."""
    # jax.debug.print appears incompatible for testing on tpu
    # so we skip the test (and we cannot just use
    # @absltest.skipIf(jax.devices()[0] == 'tpu', reason = ...))
    # because jax arrays cannot be manipulated from the top level of the python
    # program
    if jax.default_backend() in ['tpu', 'gpu']:
      return

    def fun_flat(x):
      return jnp.exp(-1 / x**2)

    w = jnp.asarray(-0.2)
    u = -jax.grad(fun_flat)(w)
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=30, verbose=True
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      _, _ = _run_linesearch(opt, fun_flat, w, u, stepsize_guess=1.0)
    self.assertIn(_linesearch.FLAG_INTERVAL_TOO_SMALL, stdout.getvalue())

  def test_failure_inf_value(self):
    """Check behavior for inf/nan values."""
    # jax.debug.print appears incompatible for testing on tpu
    # so we skip the test (and we cannot just use
    # @absltest.skipIf(jax.devices()[0] == 'tpu', reason = ...))
    # because jax arrays cannot be manipulated from the top level of the python
    # program
    if jax.default_backend() in ['tpu', 'gpu']:
      return

    def fun_inf(x):
      return jnp.log(x)

    w = jnp.asarray(1.0)
    u = jnp.asarray(-2.0)
    opt = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=30, verbose=True
    )
    _, state = _run_linesearch(opt, fun_inf, w, u, stepsize_guess=1.0)
    stepsize = optax.tree.get(state, 'learning_rate')
    self.assertGreater(stepsize, 0.0)

  def test_high_smaller_than_low(self):
    # See google/jax/issues/16236
    def fn(x):
      return x**2

    # Descent direction p chosen such that, with x+p
    # the first trial of the algorithm,
    # 1. u*f'(w) < 0 (valid descent direction)
    # 2. w+u satisfies sufficient decrease
    # 3. w+u does not satisfy small curvature
    # 4. f'(w+u) > 0
    # As a result, the first trial starts with high < low

    w = jnp.asarray(-1.0)
    u = -1.95 * w

    opt = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=20)
    _, final_state = _run_linesearch(opt, fn, w, u, stepsize_guess=1.0)
    decrease_error = optax.tree.get(final_state, 'decrease_error')
    curvature_error = optax.tree.get(final_state, 'curvature_error')
    success = (decrease_error <= 0.0) and (curvature_error <= 0.0)
    self.assertTrue(success)

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
      confuse_dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
  )
  def test_dtype_stability(self, dtype, confuse_dtype):
    kw = ['tol', 'increase_factor', 'slope_rtol', 'curv_rtol',
          'approx_dec_rtol', 'stepsize_precision']
    with utils.x64_precision(True):
      # pytype: disable=wrong-arg-types
      opt = _linesearch.scale_by_zoom_linesearch(
          max_linesearch_steps=5,
          **{k: jnp.array(1e-5, dtype=confuse_dtype) for k in kw})
      # pytype: enable=wrong-arg-types
      x = jnp.array([1.0, 2.0], dtype=dtype)
      state = opt.init(x)
      value_fn = lambda x: jnp.sum(x**2).astype(confuse_dtype)
      # TODO(rdyro): ensure optimizer updates dtype matches parameters
      cond = jax.random.randint(jax.random.key(0), (), 0, 2) == 0
      jax.lax.cond(cond, lambda x: opt.update(x, state, x, value=1.0, grad=x,
                                              value_fn=value_fn)[1],
                   lambda x: state, x)

  def test_value_and_grad_from_state(self):
    def fn(x):
      return jnp.sum(x**2)

    value_and_grad_ = _linesearch.value_and_grad_from_state(fn)

    value_and_grad = jax.jit(value_and_grad_)

    params = jnp.array([1.0, 2.0, 3.0])

    # No value and grad in this transform so it should raise an error
    opt = transform.scale_by_adam()
    state = opt.init(params)
    self.assertRaises(ValueError, value_and_grad, params, state=state)

    # Multiple values and grads in this transform so it should raise an error
    opt = combine.chain(
        _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
        _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
    )
    state = opt.init(params)
    self.assertRaises(KeyError, value_and_grad, params, state=state)

    # It should work efficiently when the linesearch stores the gradient
    opt = combine.chain(
        alias.sgd(learning_rate=1.0),
        _linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=15, store_grad=True
        ),
    )
    state = opt.init(params)
    value, grad = value_and_grad(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fn
    )
    params = update.apply_updates(params, updates)
    params = jax.block_until_ready(params)

    def false_fn(_):
      return 1.0

    false_value_and_grad_ = _linesearch.value_and_grad_from_state(false_fn)
    false_value_and_grad = jax.jit(false_value_and_grad_)

    # At the second step we should not evaluate the function
    # so in this case it should not return the output of false_fn
    value, _ = false_value_and_grad(params, state=state)
    self.assertNotEqual(value, 1.0)

  def test_extract_fns_kwargs(self):
    def fn1(a, b):
      return a + b

    def fn2(c, d):
      return c + d

    kwargs = {'b': 1.0, 'd': 2.0, 'e': 3.0}
    fns_kwargs, remaining_kwargs = _linesearch._extract_fns_kwargs(
        (fn1, fn2), kwargs
    )
    self.assertEqual(fns_kwargs, [{'b': 1.0}, {'d': 2.0}])
    self.assertEqual(remaining_kwargs, {'e': 3.0})

if __name__ == '__main__':
  absltest.main()
