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


"""Tests of gradient transformations."""

from collections.abc import Callable # For Python 3.9+
# from typing import Callable # For older Python, but collections.abc is preferred
from absl.testing import absltest
from absl.testing import parameterized
import chex
from optax._src import base # Import base for EmptyState
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update
import optax.tree

STEPS = 50
LR = 1e-2


class TransformTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  @chex.all_variants
  @parameterized.named_parameters([
      ('adadelta', transform.scale_by_adadelta),
      ('adam', transform.scale_by_adam),
      ('adamax', transform.scale_by_adamax),
      ('adan', transform.scale_by_adan),
      ('lion', transform.scale_by_lion),
      ('polyak', transform.scale_by_polyak),
      ('rmsprop', transform.scale_by_rms),
      ('stddev', transform.scale_by_stddev),
      ('trust_ratio', transform.scale_by_trust_ratio),
      ('param_block_norm', transform.scale_by_param_block_norm),
      ('param_block_rms', transform.scale_by_param_block_rms),
      ('distance_over_gradients', transform.scale_by_distance_over_gradients),
      ('normalize_by_update_norm', transform.normalize_by_update_norm),
  ])
  def test_scalers(self, scaler_constr):
    params = self.init_params

    scaler = scaler_constr()
    init_fn = self.variant(scaler.init)
    transform_fn = self.variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    if scaler_constr.__name__ == 'scale_by_polyak':
      extra_args = {'value': jnp.array(0.0)}
    else:
      extra_args = {}
    updates, state = transform_fn(
        self.per_step_updates, state, params, **extra_args
    )
    chex.assert_tree_all_finite((params, updates, state))
    jax.tree.map(lambda *args: chex.assert_equal_shape(args), params, updates)

  @chex.all_variants
  def test_apply_every(self):
    # The frequency of the application of sgd
    k = 4
    zero_update = (jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))

    # optax sgd
    optax_sgd_params = self.init_params
    sgd = alias.sgd(LR, 0.0)
    state_sgd = sgd.init(optax_sgd_params)

    # optax sgd plus apply every
    optax_sgd_apply_every_params = self.init_params
    sgd_apply_every = combine.chain(
        transform.apply_every(k=k),
        transform.trace(decay=0, nesterov=False),
        transform.scale(-LR),
    )
    state_sgd_apply_every = sgd_apply_every.init(optax_sgd_apply_every_params)
    transform_fn = self.variant(sgd_apply_every.update)

    for i in range(STEPS):
      # Apply a step of sgd
      updates_sgd, state_sgd = sgd.update(self.per_step_updates, state_sgd)
      optax_sgd_params = update.apply_updates(optax_sgd_params, updates_sgd)

      # Apply a step of sgd_apply_every
      updates_sgd_apply_every, state_sgd_apply_every = transform_fn(
          self.per_step_updates, state_sgd_apply_every
      )
      optax_sgd_apply_every_params = update.apply_updates(
          optax_sgd_apply_every_params, updates_sgd_apply_every
      )

      # Every k steps, check equivalence.
      if i % k == k - 1:
        chex.assert_trees_all_close(
            optax_sgd_apply_every_params, optax_sgd_params, atol=1e-6, rtol=1e-5
        )
      # Otherwise, check update is zero.
      else:
        chex.assert_trees_all_close(
            updates_sgd_apply_every, zero_update, atol=0.0, rtol=0.0
        )

  def test_scale(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      factor = 0.1**i
      rescaler = transform.scale(factor)
      # Apply rescaling.
      scaled_updates, _ = rescaler.update(updates, {})

      # Manually scale updates.
      def rescale(t):
        return t * factor  # pylint:disable=cell-var-from-loop  # noqa: B023

      manual_updates = jax.tree.map(rescale, updates)
      # Check the rescaled updates match.
      chex.assert_trees_all_close(scaled_updates, manual_updates)

  @parameterized.named_parameters([
      ('1d', [1.0, 2.0], [1.0, 2.0]),
      ('2d', [[1.0, 2.0], [3.0, 4.0]], [[-0.5, 0.5], [-0.5, 0.5]]),
      (
          '3d',
          [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
          [[[-1.5, -0.5], [0.5, 1.5]], [[-1.5, -0.5], [0.5, 1.5]]],
      ),
  ])
  def test_centralize(self, inputs, outputs):
    inputs = jnp.asarray(inputs)
    outputs = jnp.asarray(outputs)
    centralizer = transform.centralize()
    centralized_inputs, _ = centralizer.update(inputs, {})
    chex.assert_trees_all_close(centralized_inputs, outputs)

  def test_scale_by_optimistic_gradient(self):
    opt = transform.scale_by_optimistic_gradient()

    state = opt.init(jnp.asarray(10.0))

    grad_0 = jnp.asarray(2.0)
    opt_grad_0, state = opt.update(grad_0, state)

    grad_1 = jnp.asarray(3.0)
    opt_grad_1, state = opt.update(grad_1, state)

    grad_2 = jnp.asarray(4.0)
    opt_grad_2, _ = opt.update(grad_2, state)

    with self.subTest('Check initial update is correct'):
      # see https://github.com/google-deepmind/optax/issues/1082
      # initial step should yield 2 * grad_0 - grad_0 = grad_0
      chex.assert_trees_all_close(opt_grad_0, grad_0)

    with self.subTest('Check second update is correct'):
      chex.assert_trees_all_close(opt_grad_1, 2 * grad_1 - grad_0)

    with self.subTest('Check third update is correct'):
      chex.assert_trees_all_close(opt_grad_2, 2 * grad_2 - grad_1)

  def test_scale_by_polyak_l1_norm(self, tol=1e-10):
    """Polyak step-size on L1 norm."""
    # for this objective, the Polyak step-size has an exact model and should
    # converge to the minimizer in one step
    objective = lambda x: jnp.abs(x).sum()

    init_params = jnp.array([1.0, -1.0])
    polyak = transform.scale_by_polyak()
    polyak_state = polyak.init(init_params)
    # check that polyak state raises an error if it called without a value
    with self.assertRaises(TypeError):
      polyak.update(self.per_step_updates, polyak_state, init_params)

    value, grad = jax.value_and_grad(objective)(init_params)
    updates, _ = polyak.update(grad, polyak_state, init_params, value=value)
    # check that objective at (init_params - updates) is smaller than tol
    print(grad, value, updates)
    self.assertLess(objective(init_params - updates), tol)

  def test_rms_match_adam(self):
    """Test scale_by_rms add_eps_in_sqrt=False matches scale_by_adam(b1=0)."""
    fun = lambda x: optax.tree.norm(x, squared=True)

    rms = transform.scale_by_rms(
        decay=0.999, eps_in_sqrt=False, bias_correction=True
    )
    rms_params = self.init_params
    rms_state = rms.init(self.init_params)

    adam = transform.scale_by_adam(b1=0)
    adam_params = self.init_params
    adam_state = adam.init(self.init_params)

    for _ in range(5):
      rms_grads = jax.grad(fun)(rms_params)
      rms_updates, rms_state = rms.update(rms_grads, rms_state)
      rms_params = update.apply_updates(rms_params, rms_updates)

      adam_grads = jax.grad(fun)(adam_params)
      adam_updates, adam_state = adam.update(adam_grads, adam_state)
      adam_params = update.apply_updates(adam_params, adam_updates)

    chex.assert_trees_all_close(adam_params, rms_params)

  @chex.all_variants
  @parameterized.named_parameters([
      ('scalar_no_mask', 0.1, None),
      ('scalar_with_mask', 0.1, (True, False)),
      ('schedule_no_mask', lambda count: 0.1 * (count + 1), None),
      ('schedule_with_mask', lambda count: 0.1 * (count + 1), (True, False)),
  ])
  def test_add_decayed_weights(self, weight_decay_value_or_schedule, mask):
    params = self.init_params
    initial_updates = self.per_step_updates

    # If it's a schedule, ensure it's jit-friendly if it's a lambda
    is_lambda_schedule = (
        callable(weight_decay_value_or_schedule) and
        weight_decay_value_or_schedule.__name__ == '<lambda>')
    if is_lambda_schedule:
      # Re-define schedule_fn here for clarity in test, or pass a top-level one
      def schedule_fn(count):
        return 0.1 * (jnp.asarray(count, dtype=jnp.int32) + 1)
      current_weight_decay_source = schedule_fn
    else:
      current_weight_decay_source = weight_decay_value_or_schedule

    tx = transform.add_decayed_weights(
        weight_decay=current_weight_decay_source, mask=mask
    )
    init_fn = self.variant(tx.init)
    update_fn = self.variant(tx.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    for step in range(3):
      if callable(current_weight_decay_source):
        current_wd = current_weight_decay_source(step)
        # Verify state for schedule
        is_direct_schedule_state = isinstance(
            state, transform.ScaleByScheduleState)
        is_masked_schedule_state = (
            isinstance(state, tuple) and
            any(isinstance(s, transform.ScaleByScheduleState) for s in state))
        if is_direct_schedule_state or is_masked_schedule_state: # masked state

          actual_state_for_count = state
          # Check if state is a MaskedState tuple and extract inner state
          is_masked_tuple = (
              isinstance(state, tuple) and len(state) == 2 and
              isinstance(state[0], transform.ScaleByScheduleState))
          if is_masked_tuple: # MaskedState
             actual_state_for_count = state[0]

          if hasattr(actual_state_for_count, 'count'):
            self.assertEqual(actual_state_for_count.count, step)
      else:
        current_wd = current_weight_decay_source
        # Verify state for scalar (should be EmptyState or similar
        # for non-masked/masked)
        if mask is None:
          self.assertIsInstance(state, base.EmptyState)
        else:
          # MaskedState(inner_state=EmptyState, mask=...)
          self.assertIsInstance(state, tuple)
          self.assertIsInstance(state[0], base.EmptyState)


      # Compute expected updates
      if mask is None:
        expected_updates = jax.tree.map(
            lambda g, p, wd=current_wd: g + wd * p, initial_updates, params
        )
      else:
        # Ensure mask is a PyTree if it's a tuple for tree_map
        mask_pytree = mask
        if isinstance(mask, Callable): # If mask is a callable
            mask_pytree = mask(params)

        expected_updates = jax.tree.map(
            lambda g, p, m, wd=current_wd: g + wd * p if m else g,
            initial_updates,
            params,
            mask_pytree,
        )

      # Apply transformation
      updates, new_state = update_fn(initial_updates, state, params)
      chex.assert_tree_all_finite((params, updates, new_state))
      jax.tree.map(
          lambda *args: chex.assert_equal_shape(args), params, updates
      )
      chex.assert_trees_all_close(updates, expected_updates, atol=1e-6, rtol=1e-5)
      state = new_state # Update state for next iteration


if __name__ == '__main__':
  absltest.main()
