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
"""Tests for the Reduce on Plateau method in `reduce_on_plateau.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import test_utils
from optax.contrib import _reduce_on_plateau


class ReduceLROnPlateauTest(parameterized.TestCase):
  """Test for reduce_on_plateau scheduler."""

  def setUp(self):
    super().setUp()
    self.patience = 5
    self.cooldown = 5
    self.transform = _reduce_on_plateau.reduce_on_plateau(
        factor=0.1,
        patience=self.patience,
        rtol=1e-4,
        atol=0.0,
        cooldown=self.cooldown,
        accumulation_size=1,
        min_scale=0.01,
    )
    self.updates = {'params': jnp.array(1.0)}  # dummy updates

  def tearDown(self):
    super().tearDown()
    jax.config.update('jax_enable_x64', False)

  @parameterized.parameters(False, True)
  def test_learning_rate_reduced_after_cooldown_period_is_over(
      self, enable_x64
  ):
    """Test that learning rate is reduced after cooldown."""

    # Enable float64 if requested
    jax.config.update('jax_enable_x64', enable_x64)

    # Initialize the state
    state = self.transform.init(self.updates['params'])

    # Wait until patience runs out
    updates = self.updates
    for _ in range(self.patience + 1):
      updates, state = self.transform.update(
          updates=self.updates, state=state, value=jnp.asarray(1.0, dtype=float)
      )

    # Check that learning rate is reduced
    scale, best_value, plateau_count, cooldown_count, *_ = state
    test_utils.assert_trees_all_close(scale, 0.1)
    test_utils.assert_trees_all_close(best_value, 1.0)
    test_utils.assert_trees_all_close(plateau_count, 0)
    test_utils.assert_trees_all_close(cooldown_count, self.cooldown)
    test_utils.assert_trees_all_close(updates, {'params': jnp.array(0.1)})

    # One more step
    _, state = self.transform.update(
        updates=updates, state=state, value=jnp.asarray(1.0, dtype=float)
    )

    # Check that cooldown_count is decremented
    scale, best_value, plateau_count, cooldown_count, *_ = state
    test_utils.assert_trees_all_close(scale, 0.1)
    test_utils.assert_trees_all_close(best_value, 1.0)
    test_utils.assert_trees_all_close(plateau_count, 0)
    test_utils.assert_trees_all_close(cooldown_count, self.cooldown - 1)

  @parameterized.parameters(False, True)
  def test_learning_rate_is_not_reduced(self, enable_x64):
    """Test that plateau_count resets after a new best_value is found."""

    # Enable float64 if requested
    jax.config.update('jax_enable_x64', enable_x64)

    # State with positive plateau_count
    state = _reduce_on_plateau.ReduceLROnPlateauState(
        best_value=jnp.array(1.0),
        plateau_count=jnp.array(3, dtype=jnp.int32),
        scale=jnp.array(0.1),
        cooldown_count=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
        avg_value=jnp.array(0.0),
    )

    # Update with better value
    _, new_state = self.transform.update(
        updates=self.updates, state=state, value=jnp.asarray(0.1, float)
    )

    # Check that plateau_count resets
    scale, best_value, plateau_count, *_ = new_state
    test_utils.assert_trees_all_close(plateau_count, 0)
    test_utils.assert_trees_all_close(scale, 0.1)
    test_utils.assert_trees_all_close(best_value, 0.1)

  @parameterized.parameters(False, True)
  def test_learning_rate_not_reduced_during_cooldown(self, enable_x64):
    """Test that learning rate is not reduced during cooldown."""

    # Enable float64 if requested
    jax.config.update('jax_enable_x64', enable_x64)

    # State with positive cooldown_count
    state = _reduce_on_plateau.ReduceLROnPlateauState(
        best_value=jnp.array(1.0),
        plateau_count=jnp.array(0, dtype=jnp.int32),
        scale=jnp.array(0.1),
        cooldown_count=jnp.array(3, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
        avg_value=jnp.array(0.0),
    )

    # Update with worse value
    _, new_state = self.transform.update(
        updates=self.updates, state=state, value=jnp.asarray(2.0, dtype=float)
    )

    # Check that learning rate is not reduced and
    # plateau_count is not incremented
    scale, best_value, plateau_count, cooldown_count, *_ = new_state
    test_utils.assert_trees_all_close(scale, 0.1)
    test_utils.assert_trees_all_close(best_value, 1.0)
    test_utils.assert_trees_all_close(plateau_count, 0)
    test_utils.assert_trees_all_close(cooldown_count, 2)

  @parameterized.parameters(False, True)
  def test_learning_rate_not_reduced_after_end_scale_is_reached(
      self, enable_x64
  ):
    """Test that learning rate is not reduced if min_scale has been reached."""

    # Enable float64 if requested
    jax.config.update('jax_enable_x64', enable_x64)

    # State with scale == min_scale
    state = _reduce_on_plateau.ReduceLROnPlateauState(
        best_value=jnp.array(1.0),
        plateau_count=jnp.array(0, dtype=jnp.int32),
        scale=jnp.array(0.01),
        cooldown_count=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
        avg_value=jnp.array(0.0),
    )

    # Wait until patience runs out
    updates = self.updates
    for _ in range(self.patience + 1):
      updates, state = self.transform.update(
          updates=self.updates,
          state=state,
          value=0.1,
      )

    # Check that learning rate is not reduced
    scale, best_value, plateau_count, cooldown_count, *_ = state
    test_utils.assert_trees_all_close(scale, 0.01)
    test_utils.assert_trees_all_close(best_value, 0.1)
    test_utils.assert_trees_all_close(plateau_count, 0)
    test_utils.assert_trees_all_close(cooldown_count, self.cooldown)
    test_utils.assert_trees_all_close(updates, {'params': jnp.array(0.01)})

  @parameterized.product(
      param_dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      min_scale_dtype=(np.float32, np.float64),
      as_jax_array=(False, True),
      use_jit=(False, True),
  )
  def test_array_min_scale_preserves_state_dtype(
      self, param_dtype, min_scale_dtype, as_jax_array, use_jit
  ):
    jax.config.update('jax_enable_x64', True)
    params = {'low': jnp.ones(2, param_dtype), 'high': jnp.ones(2, jnp.float64)}
    min_scale = min_scale_dtype(0.25)
    if as_jax_array:
      min_scale = jnp.asarray(min_scale)
    transform = _reduce_on_plateau.reduce_on_plateau(
        factor=0.5,
        patience=1,
        cooldown=1,
        accumulation_size=2,
        min_scale=min_scale,
    )
    state = transform.init(params)
    initial_dtypes = jax.tree.map(lambda x: x.dtype, state)
    update = jax.jit(transform.update) if use_jit else transform.update
    expected_scales = [
        1., 1., 1., 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25
    ]
    for expected_scale in expected_scales:
      updates, state = update(params, state, value=jnp.float64(1.))
      self.assertEqual(jax.tree.map(lambda x: x.dtype, state), initial_dtypes)
      self.assertEqual(float(state.scale), expected_scale)
      for name, param in params.items():
        self.assertEqual(updates[name].dtype, param.dtype)
        np.testing.assert_allclose(updates[name], expected_scale)


if __name__ == '__main__':
  absltest.main()
