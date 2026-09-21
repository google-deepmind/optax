# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods in `factorized.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import factorized
from optax._src import test_utils
from optax.transforms import _accumulation


class FactorizedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  def test_scale_by_factored_rms(self):
    params = self.init_params

    scaler = factorized.scale_by_factored_rms()
    init_fn = jax.jit(scaler.init)
    transform_fn = jax.jit(scaler.update)

    state = init_fn(params)
    test_utils.assert_tree_all_finite(state)

    updates, state = transform_fn(self.per_step_updates, state, params)
    test_utils.assert_tree_all_finite((params, updates, state))
    test_utils.assert_trees_all_equal_shapes(params, updates)

  @parameterized.product(
      factorized_dims=(True, False), dtype=('bfloat16', 'float32')
  )
  def test_preserve_dtype(self, factorized_dims: bool, dtype: str):
    """Test that the optimizer returns updates of same dtype as params."""
    dtype = jnp.dtype(dtype)
    opt = factorized.scale_by_factored_rms()
    fun = lambda x: jnp.sum(x**2)

    if factorized_dims:
      # The updates are factored only for large enough parameters
      # default min_dim_size_to_factor is 128 so we use 129 here.
      params = jnp.ones((129, 129), dtype=dtype)
    else:
      params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = jax.jit(opt.init)(params)
    updates, _ = jax.jit(opt.update)(grads, state, params)
    self.assertEqual(updates.dtype, params.dtype)

  @parameterized.product(
      factorized_dims=(True, False), dtype=('bfloat16', 'float32')
  )
  def test_gradient_accumulation(self, factorized_dims, dtype):
    """Test that the optimizers can safely be used with optax.MultiSteps."""
    # Checks if https://github.com/google-deepmind/optax/issues/377 is fixed.
    dtype = jnp.dtype(dtype)
    base_opt = factorized.scale_by_factored_rms()
    opt = _accumulation.MultiSteps(base_opt, every_k_schedule=4)

    fun = lambda x: jnp.sum(x**2)

    if factorized_dims:
      # The updates are factored only for large enough parameters
      # default min_dim_size_to_factor is 128 so we use 129 here.
      params = jnp.ones((129, 129), dtype=dtype)
    else:
      params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = jax.jit(opt.init)(params)
    updates, _ = jax.jit(opt.update)(grads, state, params)
    test_utils.assert_trees_all_equal(updates, jnp.zeros_like(grads))

  @parameterized.product(step_offset=(1, 3), factored_dims=(True, False))
  def test_step_offset_keeps_the_state_finite(
      self, step_offset, factored_dims
  ):
    """A positive step_offset must not make the second moment non-finite.

    `count` starts at zero on a fresh init, so the local step is negative
    until the offset is passed. The decay schedule used to be evaluated
    there, giving -inf at zero and nan below it, and nan in `v` survives
    every later update, so the run never recovered.
    """
    scaler = factorized.scale_by_factored_rms(step_offset=step_offset)
    if factored_dims:
      # The updates are factored only for large enough parameters
      # default min_dim_size_to_factor is 128 so we use 129 here.
      params = jnp.ones((129, 129))
    else:
      params = jnp.array([1.0, 2.0])
    state = jax.jit(scaler.init)(params)
    update_fn = jax.jit(scaler.update)

    for i in range(step_offset + 2):
      grads = jnp.full(params.shape, 0.1 * (i + 1))
      updates, state = update_fn(grads, state, params)
      test_utils.assert_tree_all_finite((updates, state))

  def test_step_offset_holds_the_schedule_at_its_first_step(self):
    """Before the offset, the decay rate is the one for local step 0."""
    step_offset = 3
    epsilon = 1e-30
    scaler = factorized.scale_by_factored_rms(
        step_offset=step_offset, min_dim_size_to_factor=128, epsilon=epsilon
    )
    params = jnp.array([1.0, 2.0])
    state = scaler.init(params)

    # The rate at local step 0 is 0.0, so `v` is the latest squared
    # gradient exactly.
    for i in range(step_offset + 1):
      grads = jnp.full(params.shape, 0.1 * (i + 1))
      _, state = scaler.update(grads, state, params)
      test_utils.assert_trees_all_close(state.v, grads**2 + epsilon)

    # One step past the offset the schedule decays and `v` lags behind.
    grads = jnp.full(params.shape, 5.0)
    _, state = scaler.update(grads, state, params)
    self.assertTrue(bool(jnp.all(state.v < grads**2)))

  def test_step_offset_restarts_a_restored_schedule(self):
    """A restored count plus a matching offset starts the schedule over."""
    step_offset = 100
    scaler = factorized.scale_by_factored_rms(step_offset=step_offset)
    fresh = factorized.scale_by_factored_rms()
    params = jnp.array([1.0, 2.0])
    grads = jnp.array([0.5, 0.25])

    restored = scaler.init(params)
    restored = restored._replace(count=jnp.asarray(step_offset, jnp.int32))
    restored_updates, _ = scaler.update(grads, restored, params)
    fresh_updates, _ = fresh.update(grads, fresh.init(params), params)

    test_utils.assert_trees_all_close(restored_updates, fresh_updates)


if __name__ == '__main__':
  absltest.main()
