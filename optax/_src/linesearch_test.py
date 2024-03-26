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
"""Tests for `linesearch.py`."""

import functools
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from optax._src import alias
from optax._src import combine
from optax._src import linesearch
from optax._src import update
from optax._src import utils
import optax.tree_utils as optax_tu


class BacktrackingLinesearchTest(chex.TestCase):

  def get_fun(self, name):
    """Common ill-behaved functions."""

    def rosenbrock(x):
      return jnp.sum(100.0 * jnp.diff(x) ** 2 + (1.0 - x[:-1]) ** 2)

    def himmelblau(x):
      return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2

    def matyas(x):
      return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

    def eggholder(x):
      return -(x[1] + 47) * jnp.sin(
          jnp.sqrt(jnp.abs(x[0] / 2.0 + x[1] + 47.0))
      ) - x[0] * jnp.sin(jnp.sqrt(jnp.abs(x[0] - (x[1] + 47.0))))

    funs = dict(
        rosenbrock=rosenbrock,
        himmelblau=himmelblau,
        matyas=matyas,
        eggholder=eggholder,
    )
    return funs[name]

  def check_decrease_conditions(
      self, fun, init_params, descent_dir, final_params, final_state, opt_args
  ):
    """Check decrease conditions."""
    init_value, init_grad = jax.value_and_grad(fun)(init_params)
    final_value = fun(final_params)
    final_lr = final_state[0]

    slope = optax_tu.tree_vdot(descent_dir, init_grad)
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

  @chex.all_variants
  @parameterized.product(
      name_fun_and_init_params=[
          ('rosenbrock', np.zeros(2)),
          ('himmelblau', np.ones(2)),
          ('matyas', np.ones(2) * 6.0),
          ('eggholder', np.ones(2) * 100.0),
      ],
      increase_factor=[1.0, 1.5, math.inf],
      slope_rtol=[1e-4, 0.0],
      atol=[1e-4, 0.0],
      rtol=[1e-4, 0.0],
  )
  def test_linesearch_one_step(
      self,
      name_fun_and_init_params,
      increase_factor,
      slope_rtol,
      atol,
      rtol,
  ):
    name_fun, init_params = name_fun_and_init_params
    fn = self.get_fun(name_fun)
    base_opt = alias.sgd(learning_rate=1.0)
    descent_dir = -jax.grad(fn)(init_params)

    opt_args = dict(
        max_backtracking_steps=30,
        slope_rtol=slope_rtol,
        increase_factor=increase_factor,
        atol=atol,
        rtol=rtol,
    )

    solver = combine.chain(
        base_opt,
        linesearch.scale_by_backtracking_linesearch(**opt_args),
    )
    init_state = solver.init(init_params)

    update_fn = functools.partial(solver.update, value_fn=fn)
    update_fn = self.variant(update_fn)
    value, grad = jax.value_and_grad(fn)(init_params)
    updates, state = update_fn(
        grad, init_state, init_params, value=value, grad=grad
    )
    params = update.apply_updates(init_params, updates)

    self.check_decrease_conditions(
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
        linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
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
    chex.assert_trees_all_close(final_params, params, atol=1e-2, rtol=1e-2)

  @chex.variants(
      with_jit=True,
      without_jit=True,
      with_pmap=False,
      with_device=True,
      without_device=True,
  )
  def test_recycling_value_and_grad(self):
    # A vmap or a pmap makes the cond in value_and_state_from_grad
    # become a select and in that case this code cannot be optimal.
    # So we skip the pmap test.
    init_params = jnp.array([1.0, 10.0, 1.0])
    final_params = jnp.array([1.0, -1.0, 1.0])

    def fn(params):
      return jnp.sum((params - final_params) ** 2)

    value_and_grad = utils.value_and_grad_from_state(fn)

    base_opt = alias.sgd(learning_rate=0.1)
    solver = combine.chain(
        base_opt,
        linesearch.scale_by_backtracking_linesearch(
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

    fake_value_and_grad = utils.value_and_grad_from_state(fake_fun)

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

    step = self.variant(step_)
    params = init_params
    state = init_state
    for iter_num in range(max_iter):
      params, state = step(params, state, iter_num)
    params = jax.block_until_ready(params)
    chex.assert_trees_all_close(final_params, params, atol=1e-2, rtol=1e-2)

  def test_armijo_sgd(self):
    def fn(params, x, y):
      return jnp.sum((x.dot(params) - y) ** 2)

    # Create artificial data
    noise = 1e-3
    key = jrd.PRNGKey(0)
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
        linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
    )
    num_passes = 10

    state = solver.init(init_params)
    update_fn = functools.partial(solver.update, value_fn=fn)
    update_fn = jax.jit(update_fn)

    params = init_params
    for _ in range(num_passes):
      x, y = next(xs_iter), next(ys_iter)
      value, grad = jax.value_and_grad(fn)(params, x, y)
      updates, state = update_fn(
          grad, state, params, value=value, grad=grad, x=x, y=y
      )
      params = update.apply_updates(params, updates)
    chex.assert_trees_all_close(
        params, target_params, atol=5 * 1e-2, rtol=5 * 1e-2
    )


if __name__ == '__main__':
  absltest.main()
