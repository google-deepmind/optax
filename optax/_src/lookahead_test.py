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
"""Tests for the lookahead optimizer in `lookahead.py`."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import lookahead
from optax._src import test_utils
from optax._src import update
from optax.transforms import _conditionality
import optax.tree


def _build_sgd():
  return alias.sgd(1.0)


class OptimizerTestState(NamedTuple):
  """Fast optimizer state for the lookahead tests."""

  aggregate_grads: base.Params
  # Include a variable with non-zero initial value to check that it is reset
  # correctly by the lookahead optimizer.
  is_reset: bool = True


def _test_optimizer(step_size: float) -> base.GradientTransformation:
  """Fast optimizer for the lookahead tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behavior of lookahead can be tested.
  def init_fn(params):
    aggregate_grads = jax.tree.map(jnp.zeros_like, params)
    return OptimizerTestState(aggregate_grads, is_reset=True)

  def update_fn(updates, state, params):
    # The test optimizer does not use the parameters, but we check that they
    # have been passed correctly.
    test_utils.assert_trees_all_equal_shapes(updates, params)
    aggregate_grads = update.apply_updates(state.aggregate_grads, updates)
    updates = jax.tree.map(lambda u: step_size * u, updates)
    return updates, OptimizerTestState(aggregate_grads, is_reset=False)

  return base.GradientTransformation(init_fn, update_fn)


class LookaheadTest(parameterized.TestCase):
  """Tests for the lookahead optimizer."""

  def setUp(self):
    super().setUp()
    self.grads = {'x': np.array(2.0), 'y': np.array(-2.0)}
    self.initial_params = {'x': np.array(3.0), 'y': np.array(-3.0)}
    self.synced_initial_params = lookahead.LookaheadParams.init_synced(
        self.initial_params
    )

  def loop(self, optimizer, num_steps, params):
    """Performs a given number of optimizer steps."""
    init_fn, update_fn = optimizer
    step = jax.jit(update_fn)
    opt_state = jax.jit(init_fn)(params)

    # A no-op change, to verify that tree map works.
    opt_state = optax.tree.map_params(init_fn, lambda v: v, opt_state)

    for _ in range(num_steps):
      updates, opt_state = step(self.grads, opt_state, params)
      params = update.apply_updates(params, updates)

    return params, opt_state

  def test_lookahead(self):
    """Tests the lookahead optimizer in an analytically tractable setting."""
    sync_period = 3
    optimizer = lookahead.lookahead(
        _test_optimizer(-0.5), sync_period=sync_period, slow_step_size=1 / 3
    )

    final_params, _ = self.loop(
        optimizer, 2 * sync_period, self.synced_initial_params
    )
    # x steps must be: 3 -> 2 -> 1 -> 2 (sync) -> 1 -> 0 -> 1 (sync).
    # Similarly for y (with sign flipped).
    correct_final_params = {'x': 1, 'y': -1}
    test_utils.assert_trees_all_close(final_params.slow, correct_final_params)

  @parameterized.parameters([False], [True])
  def test_lookahead_state_reset(self, reset_state):
    """Checks that lookahead resets the fast optimizer state correctly."""
    num_steps = sync_period = 3
    fast_optimizer = _test_optimizer(-0.5)
    optimizer = lookahead.lookahead(
        fast_optimizer,
        sync_period=sync_period,
        slow_step_size=0.5,
        reset_state=reset_state,
    )

    _, opt_state = self.loop(optimizer, num_steps, self.synced_initial_params)

    # A no-op change, to verify that this does not break anything
    opt_state = optax.tree.map_params(optimizer, lambda v: v, opt_state)

    fast_state = opt_state.fast_state
    if reset_state:
      correct_state = fast_optimizer.init(self.initial_params)
    else:
      _, correct_state = self.loop(
          fast_optimizer, num_steps, self.initial_params
      )

    test_utils.assert_trees_all_close(fast_state, correct_state)
    self.assertEqual(
        jnp.asarray(fast_state.is_reset).dtype,
        jnp.asarray(correct_state.is_reset).dtype,
    )

  @parameterized.product(reset_state=[False, True], sync_period=[1, 3])
  def test_lookahead_apply_if_finite_scan(self, reset_state, sync_period):
    fast_optimizer = _conditionality.apply_if_finite(
        alias.sgd(0.1, momentum=0.9), max_consecutive_errors=2
    )
    optimizer = lookahead.lookahead(
        fast_optimizer, sync_period, slow_step_size=0.5, reset_state=reset_state
    )
    initial_params = lookahead.LookaheadParams.init_synced(
        jnp.array([1.0, -1.0])
    )
    grads = jnp.array(
        [
            [1.0, -1.0],
            [jnp.nan, 0.0],
            [2.0, 1.0],
            [jnp.inf, 1.0],
            [3.0, -2.0],
            [4.0, 2.0],
        ]
    )

    def step(carry, grad):
      params, state = carry
      updates, state = optimizer.update(grad, state, params)
      params = update.apply_updates(params, updates)
      return (params, state), (params, state)

    _, (actual_params, actual_states) = jax.lax.scan(
        step, (initial_params, optimizer.init(initial_params)), grads
    )

    # Reference the two Lookahead phases directly, without arithmetic selection
    # of either parameters or state. Rejected gradients also exercise the bool
    # flag and integer counters of the stock apply_if_finite wrapper.
    fast_params = slow_params = initial_params.fast
    fast_state = fast_optimizer.init(fast_params)
    expected_params = []
    expected_states = []
    for i, grad in enumerate(grads):
      updates, fast_state = fast_optimizer.update(grad, fast_state, fast_params)
      next_fast_params = update.apply_updates(fast_params, updates)
      if (i + 1) % sync_period == 0:
        slow_params = slow_params + 0.5 * (next_fast_params - slow_params)
        next_fast_params = slow_params
        if reset_state:
          fast_state = fast_optimizer.init(fast_params)
      fast_params = next_fast_params
      expected_params.append(
          lookahead.LookaheadParams(fast_params, slow_params)
      )
      expected_states.append(fast_state)

    expected_params = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), *expected_params
    )
    expected_states = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), *expected_states
    )
    test_utils.assert_trees_all_close(actual_params, expected_params)
    test_utils.assert_trees_all_close(actual_states.fast_state, expected_states)
    for actual, expected in zip(
        jax.tree.leaves(actual_states.fast_state),
        jax.tree.leaves(expected_states),
    ):
      self.assertEqual(actual.dtype, expected.dtype)

  @parameterized.product(
      sync_period=[1, 3],
      state_values=[
          (0.0, jnp.inf),
          (0.0, jnp.nan),
          (jnp.inf, 1.0),
          (jnp.nan, 1.0),
      ],
  )
  def test_lookahead_reset_selects_nonfinite_state(
      self, sync_period, state_values
  ):
    initial_value, updated_value = state_values

    def init_fn(params):
      return {
          'count': jnp.array(0, jnp.int32),
          'value': jnp.full_like(params, initial_value),
      }

    def update_fn(updates, state, params=None):
      return updates, {
          'count': state['count'] + 1,
          'value': jnp.full_like(state['value'], updated_value),
      }

    optimizer = lookahead.lookahead(
        base.GradientTransformation(init_fn, update_fn),
        sync_period=sync_period,
        slow_step_size=0.5,
        reset_state=True,
    )
    params = lookahead.LookaheadParams.init_synced(jnp.array([1.0, -1.0]))
    state = optimizer.init(params)
    step = jax.jit(optimizer.update)
    for i in range(sync_period):
      _, state = step(jnp.zeros_like(params.fast), state, params)
      synced = i + 1 == sync_period
      expected = {
          'count': jnp.array(0 if synced else i + 1, jnp.int32),
          'value': jnp.full_like(
              params.fast, initial_value if synced else updated_value
          ),
      }
      for actual, wanted in zip(
          jax.tree.leaves(state.fast_state), jax.tree.leaves(expected)
      ):
        self.assertEqual(actual.dtype, wanted.dtype)
        np.testing.assert_array_equal(actual, wanted)

  @parameterized.parameters(
      [1, 0.5, {'x': np.array(1.), 'y': np.array(-1.)}],
      [1, 0, {'x': np.array(3.), 'y': np.array(-3.)}],
      [1, 1, {'x': np.array(-1.), 'y': np.array(1.)}],
      [2, 1, {'x': np.array(-1.), 'y': np.array(1.)}])  # pyformat: disable
  def test_lookahead_edge_cases(
      self, sync_period, slow_step_size, correct_result
  ):
    """Checks special cases of the lookahed optimizer parameters."""
    # These edge cases are important to check since users might use them as
    # simple ways of disabling lookahead in experiments.
    optimizer = lookahead.lookahead(
        _test_optimizer(-1), sync_period, slow_step_size
    )
    final_params, _ = self.loop(
        optimizer, num_steps=2, params=self.synced_initial_params
    )
    test_utils.assert_trees_all_close(final_params.slow, correct_result)


if __name__ == '__main__':
  absltest.main()
