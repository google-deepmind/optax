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
"""Tests for extra_kwargs."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import transform
from optax._src.experimental import extra_args as extra
from optax._src.experimental.extra_args import  ReduceLROnPlateauState,reduce_on_plateau


def scale_by_loss():
  """Scale the gradient by the absolute value of the loss."""

  def init_fn(params, *, extra_args):
    del params, extra_args
    return base.EmptyState()

  def update_fn(updates, state, params, *, extra_args):
    del params
    updates = jax.tree_util.tree_map(
        lambda u: u / extra_args['loss'], updates)
    return updates, state

  return extra.GradientTransformationWithExtraArgs(init_fn, update_fn)


class ExtraArgsTest(absltest.TestCase):

  def test_named_chain(self):
    tx = extra.named_chain(
        ('scale', transform.scale(0.1)),
        ('scale_by_policy_loss', scale_by_loss()),
        ('scale_by_value_loss', scale_by_loss()),
    )

    params = {'a': jnp.ones((4,))}
    grads = params
    extra_args = {
        'scale_by_policy_loss': {'loss': 0.01},
        'scale_by_value_loss': {'loss': 10.0}}

    opt_state = tx.init(params, extra_args=extra_args)
    updates, opt_state = tx.update(
        grads, opt_state, params, extra_args=extra_args)
    chex.assert_trees_all_close(updates, {'a': jnp.ones((4,))})

class ReduceLROnPlateauTest(absltest.TestCase):
    def test_learning_rate_reduced_after_cooldown_period_is_over(self):
        """Test that learning rate is reduced again after cooldown period is over."""
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
        transform = reduce_on_plateau(reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
        new_updates, new_state = transform.update(updates=updates, state=state, extra_args=extra_args)
        # Check that learning rate is reduced
        assert new_state.lr == 0.005
        assert new_state.plateau_count == 0
        assert new_state.cooldown_counter == 5
        new_updates, new_state = transform.update(updates=updates, state=new_state, extra_args=extra_args)
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
        transform = reduce_on_plateau(reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
        new_updates, new_state = transform.update(updates=updates, state=state, extra_args=extra_args)
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
        transform = reduce_on_plateau(reduce_factor=0.5, patience=5, min_improvement=1e-4, cooldown=5)
        new_updates, new_state = transform.update(updates=updates, state=state, extra_args=extra_args)
        # Check that learning rate is not reduced
        assert new_state.lr == 0.01

    
if __name__ == '__main__':
  absltest.main()
