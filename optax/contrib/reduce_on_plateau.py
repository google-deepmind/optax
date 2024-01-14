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
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics


class ReduceLROnPlateauState(NamedTuple):
  """State for the ReduceLROnPlateau callback."""

  lr: chex.Array  # shape=(), dtype=jnp.float32
  best_loss: chex.Array  # shape=(), dtype=jnp.float32
  plateau_count: chex.Array  # shape=(), dtype=jnp.int32
  cooldown_counter: chex.Array  # shape=(), dtype=jnp.int32


def reduce_on_plateau(
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
    cooldown: int = 0,
) -> base.GradientTransformationExtraArgs:
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning once learning stagnates.
  his scheduler reads a metrics quantity and if no improvement is seen for
  a ``patience`` number of epochs, the learning rate is reduced.

  Args:
    factor: Factor by which to reduce the learning rate. new_lr = lr * factor.
    patience: Number of iterations with no improvement after which learning rate
      will be reduced.
    threshold: Threshold for measuring the new optimum, to only focus on
      significant changes.
    cooldown: Number of iterations to wait before resuming normal operation
      after lr has been reduced.

  Returns:
    A GradientTransformationExtraArgs object.
  """

  def init_fn(params) -> ReduceLROnPlateauState:
    del params
    return ReduceLROnPlateauState(
        best_loss=jnp.asarray(float("inf"), dtype=jnp.float32),
        plateau_count=jnp.asarray(0, jnp.int32),
        lr=jnp.asarray(1.0, dtype=jnp.float32),
        cooldown_counter=jnp.asarray(0, jnp.int32),
    )

  def update_fn(
      updates: base.Updates,
      state: ReduceLROnPlateauState,
      params=None,
      *,
      loss,
      **extra_args,
  ) -> Tuple[base.Params, ReduceLROnPlateauState]:
    del params, extra_args

    # Update plateau count and check if plateaued
    has_improved = jnp.where((loss / state.best_loss - 1) < -threshold, 1, 0)
    new_best_loss = jnp.where(has_improved, loss, state.best_loss)

    curr_plateau_count = jnp.where(
        has_improved, 0, numerics.safe_int32_increment(state.plateau_count)
    )

    # We're in cooldown, so reduce the counter and ignore any bad epochs
    def in_cooldown():
      new_plateau_count = jnp.asarray(0, jnp.int32)
      new_lr = state.lr
      new_cooldown_counter = state.cooldown_counter - 1
      return new_plateau_count, new_lr, new_cooldown_counter

    # We're not in cooldown, so update the plateau count and lr as usual
    def not_in_cooldown():
      new_plateau_count = jnp.where(
          curr_plateau_count == patience, 0, curr_plateau_count
      )
      new_lr = jnp.where(
          curr_plateau_count == patience,
          state.lr * factor,
          state.lr,
      )
      new_cooldown_counter = jnp.where(
          curr_plateau_count == patience, cooldown, 0
      ).astype(jnp.int32)
      return new_plateau_count, new_lr, new_cooldown_counter

    new_plateau_count, new_lr, new_cooldown_counter = jax.lax.cond(
        state.cooldown_counter > 0, in_cooldown, not_in_cooldown
    )

    updates = jax.tree_util.tree_map(lambda g: new_lr * g, updates)

    new_state = ReduceLROnPlateauState(
        plateau_count=new_plateau_count,
        best_loss=new_best_loss,
        lr=new_lr,
        cooldown_counter=new_cooldown_counter,
    )
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
