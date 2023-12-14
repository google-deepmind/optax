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
"""Reduce Learning Rate on Plateau callback.

This callback monitors a quantity and if no improvement is seen for a 'patience'
number of epochs, the learning rate is reduced by a factor of 'reduce_factor'.
Optionally, a cooldown period can be specified during which the learning rate
will not be reduced.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from optax._src import base


class ReduceLROnPlateauState(NamedTuple):
  """State for the ReduceLROnPlateau callback."""
  reduce_factor: float
  patience: int
  min_improvement: float
  best_loss: float
  plateau_count: int
  lr: float
  cooldown_counter: int
  cooldown: int


def reduce_on_plateau(
  reduce_factor: float,
  patience: int,
  min_improvement: float,
  cooldown: int
) -> base.GradientTransformationExtraArgs:
  """ Reduce learning rate when a metric has stopped improving. 

  Models often benefit from reducing the learning once learning stagnates.
  his scheduler reads a metrics quantity and if no improvement is seen for
  a ‘patience’ number of epochs, the learning rate is reduced.

  Args:

  reduce_factor: Factor by which the learning rate will be reduced. 
      new_lr = lr * factor.
  patience: Number of iterations with no improvement after which learning 
      rate will be reduced.
  min_improvement: Threshold for measuring the new optimum, to only focus on 
      significant changes.
  cooldown: Number of iterations to wait before resuming normal operation 
      after lr has been reduced.
  """


  def init_fn(params):
    del params
    return ReduceLROnPlateauState(patience=patience,
                                  reduce_factor=reduce_factor,
                                  min_improvement=min_improvement,
                                  cooldown=cooldown,
                                  cooldown_counter=0,
                                  plateau_count=0,
                                  best_loss=float("inf"),
                                  lr=1,
                                  )

  def update_fn(
    updates,
    state,
    params=None,
    extra_args=None,
  ):
    del params
    if extra_args is None:
      extra_args = {}
    current_loss = extra_args.get("loss")

    # Check if the current loss is the best so far
    best_loss = state.best_loss
    # Update plateau count and check if plateaued
    has_improved = jnp.where(
      (current_loss / best_loss - 1) < -state.min_improvement, 1, 0
    )
    new_best_loss = jnp.where(has_improved, current_loss, best_loss)

    curr_plateau_count = jnp.where(has_improved, 0, state.plateau_count + 1)


    # We're in cooldown, so reduce the counter and ignore any bad epochs
    def in_cooldown():
      new_plateau_count = 0
      new_lr = state.lr
      new_cooldown_counter = state.cooldown_counter - 1
      return new_plateau_count, new_lr, new_cooldown_counter

    # We're not in cooldown, so update the plateau count and lr as usual
    def not_in_cooldown():
      new_plateau_count = jnp.where(
        curr_plateau_count == state.patience, 0, curr_plateau_count
      )
      new_lr = jnp.where(
        curr_plateau_count == state.patience,
        state.lr * state.reduce_factor,
        state.lr,
      )
      new_cooldown_counter = jnp.where(
        curr_plateau_count == state.patience, state.cooldown, 0
      )
      return new_plateau_count, new_lr, new_cooldown_counter

    new_plateau_count, new_lr, new_cooldown_counter = jax.lax.cond(
      state.cooldown_counter > 0, in_cooldown, not_in_cooldown)

    updates = jax.tree_util.tree_map(lambda g: new_lr * g, updates)

    new_state = ReduceLROnPlateauState(
      patience=state.patience,
      reduce_factor=state.reduce_factor,
      min_improvement=state.min_improvement,
      plateau_count=new_plateau_count,
      best_loss=new_best_loss,
      lr=new_lr,
      cooldown_counter=new_cooldown_counter,
      cooldown=state.cooldown,
    )
    return updates, new_state


  return base.GradientTransformationExtraArgs(init_fn, update_fn)
