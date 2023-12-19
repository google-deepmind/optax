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
"""Tests for `reduce_on_plateau.py`."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
from optax import contrib


class ReduceLROnPlateauTest(absltest.TestCase):

  def test_learning_rate_reduced_after_cooldown_period_is_over(self):
    """Test that learning rate is reduced again after cooldown period is over."""

    # Define a dummy update and extra_args
    updates = {'params': jnp.array(1.0)}
    patience = 5
    cooldown = 5
    # Apply the transformation to the updates and state
    transform = contrib.reduce_on_plateau(patience=patience, cooldown=cooldown)
    state = transform.init(updates['params'])
    for _ in range(patience + 1):
      updates, state = transform.update(updates=updates, state=state, loss=1.0)
    # Check that learning rate is reduced
    # we access the fields inside new_state using indices instead of attributes
    # because otherwise pytype throws an error
    lr, best_loss, plateau_count, cooldown_counter = state
    chex.assert_trees_all_close(lr, 0.1)
    chex.assert_trees_all_close(best_loss, 1.0)
    chex.assert_trees_all_close(plateau_count, 0)
    chex.assert_trees_all_close(cooldown_counter, cooldown)
    chex.assert_trees_all_close(updates, {'params': jnp.array(0.1)})

    _, state = transform.update(updates=updates, state=state, loss=1.0)
    lr, best_loss, plateau_count, cooldown_counter = state
    chex.assert_trees_all_close(lr, 0.1)
    chex.assert_trees_all_close(best_loss, 1.0)
    chex.assert_trees_all_close(plateau_count, 0)
    chex.assert_trees_all_close(cooldown_counter, cooldown - 1)

  def test_learning_rate_is_not_reduced(self):
    """Test that plateau count resets after a new best loss is found."""
    state = contrib.ReduceLROnPlateauState(
        best_loss=jnp.array(0.1, dtype=jnp.float32),
        plateau_count=jnp.array(3, dtype=jnp.int32),
        lr=jnp.array(0.01, dtype=jnp.float32),
        cooldown_counter=jnp.array(0, dtype=jnp.int32),
    )
    # Define a dummy update and extra_args
    updates = {'params': 1}
    # Apply the transformation to the updates and state
    transform = contrib.reduce_on_plateau(
        factor=0.5, patience=5, threshold=1e-4, cooldown=5
    )
    _, new_state = transform.update(updates=updates, state=state, loss=0.01)
    lr, best_loss, plateau_count, _ = new_state
    # Check that plateau count resets
    chex.assert_trees_all_close(plateau_count, 0)
    chex.assert_trees_all_close(lr, 0.01)
    chex.assert_trees_all_close(best_loss, 0.01)

  def test_learning_rate_not_reduced_during_cooldown(self):
    """Test that learning rate is not reduced during cooldown."""
    # Define a state where cooldown_counter is positive
    state = contrib.ReduceLROnPlateauState(
        best_loss=jnp.array(0.1, dtype=jnp.float32),
        plateau_count=jnp.array(4, dtype=jnp.int32),
        lr=jnp.array(0.01, dtype=jnp.float32),
        cooldown_counter=jnp.array(3, dtype=jnp.int32),
    )
    # Define a dummy update and extra_args
    updates = {'params': 1}
    # Apply the transformation to the updates and state
    transform = contrib.reduce_on_plateau(
        factor=0.5, patience=5, threshold=1e-4, cooldown=5
    )
    _, new_state = transform.update(updates=updates, state=state, loss=0.15)
    # Check that learning rate is not reduced
    lr, _, _, _ = new_state
    chex.assert_trees_all_close(lr, 0.01)


if __name__ == '__main__':
  absltest.main()
