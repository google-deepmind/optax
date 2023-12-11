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

from optax.contrib.reduce_on_plateau import  ReduceLROnPlateauState, reduce_on_plateau


class ReduceLROnPlateauTest(absltest.TestCase):
  def test_learning_rate_reduced_after_cooldown_period_is_over(self):
    """Test that learning rate is reduced again after cooldown period is over.
    """
    # Define a state where cooldown_counter is zero
    state = ReduceLROnPlateauState(
      reduce_factor=0.5,
      patience=5,
      min_improvement=1e-4,
      best_loss=0.1,
      ##
      plateau_count=4,
      ##
      lr=0.01,
      cooldown=5,
      cooldown_counter=0,
    )
    # Define a dummy update and extra_args
    updates = {'params': 1}
    extra_args = {'loss': 0.15}
    # Apply the transformation to the updates and state
    transform = reduce_on_plateau(
      reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
    _, new_state = transform.update(
      updates=updates, state=state, extra_args=extra_args)
    # Check that learning rate is reduced
    assert new_state.lr == 0.005
    assert new_state.plateau_count == 0
    assert new_state.cooldown_counter == 5
    _, new_state = transform.update(
      updates=updates, state=new_state, extra_args=extra_args)
    assert new_state.lr == 0.005
    assert new_state.plateau_count == 0
    assert new_state.cooldown_counter == 4


  def test_learning_rate_is_not_reduced(self):
    """Test that plateau count resets after a new best loss is found."""
    state = ReduceLROnPlateauState(
      reduce_factor=0.5,
      patience=5,
      min_improvement=1e-4,
      best_loss=0.1,
      plateau_count=3,
      lr=0.01,
      cooldown_counter=0,
      cooldown=5,
    )
    # Define a dummy update and extra_args
    updates = {'params': 1}
    extra_args = {'loss': 0.01}
    # Apply the transformation to the updates and state
    transform = reduce_on_plateau(
      reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
    _, new_state = transform.update(
      updates=updates, state=state, extra_args=extra_args)
    # Check that plateau count resets
    assert new_state.plateau_count == 0
    assert new_state.best_loss == 0.01


  def test_learning_rate_not_reduced_during_cooldown(self):
    """Test that learning rate is not reduced during cooldown."""
    # Define a state where cooldown_counter is positive
    state = ReduceLROnPlateauState(
      reduce_factor=0.5,
      patience=5,
      min_improvement=1e-4,
      best_loss=0.1,
      plateau_count=4,
      lr=0.01,
      cooldown=5,
      cooldown_counter=3,
    )
    # Define a dummy update and extra_args
    updates = {'params': 1}
    extra_args = {'loss': 0.15}
    # Apply the transformation to the updates and state
    transform = reduce_on_plateau(
      reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
    _, new_state = transform.update(
      updates=updates, state=state, extra_args=extra_args)
    # Check that learning rate is not reduced
    assert new_state.lr == 0.01


if __name__ == '__main__':
  absltest.main()
  