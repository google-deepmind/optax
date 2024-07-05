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
"""Schedule-Free wrapper for faster training & removes the need for lr decay."""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import base


class ScheduleFreeState(NamedTuple):
  """State for schedule_free."""

  b1: chex.Array
  weight_sum: chex.Array
  step_count: chex.Array
  max_lr: chex.Array
  base_optimizer_state: base.OptState
  z: base.Params


def schedule_free_eval_params(state: ScheduleFreeState, params: base.Params):
  """Params for evaluation of :func:`optax.contrib.schedule_free`."""
  return jax.tree_util.tree_map(
      lambda yi, zi: (yi - (1.0 - state.b1) * zi) / state.b1, params, state.z
  )


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    weight_lr_power: float = 2.0,
    state_dtype=jnp.float32,
) -> base.GradientTransformationExtraArgs:
  r"""Turn base_optimizer schedule_free.

  Accumulates updates returned by the base_optimizer w/o Momentum and
  replaces the momentum of an underlying optimizer with a combination of
  interpolation and averaging. In the case of gradient descent the update is

  .. math::

    \begin{align*}
      y_{t} & = (1-\beta_1)z_{t} + \beta_1 x_{t},\\
      z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
      x_{t+1} & =\left(1-\frac{1}{t}\right)x_{t}+\frac{1}{t}z_{t+1},
    \end{align*}

  Here :math:`x` is the sequence that evaluations of test/val loss should occur
  at,  which differs from the primary iterates :math:`z` and the gradient
  evaluation locations :math:`y`. The updates to :math:`z` correspond to the
  underlying optimizer, in this case a simple gradient step. Note that,
  :math:`\beta_1` corresponds to `b1` in the code.

  As the name suggests, Schedule-Free learning does not require a decreasing
  learning rate schedule, yet typically out-performs, or at worst matches, SOTA
  schedules such as cosine-decay and linear decay. Only two sequences need to be
  stored at a time (the third can be computed from the other two on the fly) so
  this method has the same memory requirements as the base optimizer (parameter
  buffer + momentum).

  In practice, authors recommend tuning :math:`\beta_1`, `warmup_steps` and
  `peak_lr` for each problem seperately. Default for :math:`\beta_1` is 0.9 but
  `0.95` and `0.98` may also work well. Schedule-Free can be wrapped on top of
  any optax optimizer. At test time, the parameters should be evaluated using
  :func:`optax.contrib.schedule_free_eval_params` as presented below.

  For example, change this::

    learning_rate_fn = optax.warmup_cosine_decay_schedule(peak_value=tuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=b1)

  To::

    learning_rate_fn = optax.warmup_constant_schedule(peak_value=retuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=0.)
    optimizer = optax.contrib.schedule_free(optimizer, learning_rate_fn, b1=b1)
    ..
    params_for_eval = optax.contrib.schedule_free_eval_params(state, params)

  Especially note that is important to switch off Momentum of the base
  optimizer. As of Apr, 2024, schedule_free is tested with SGD and Adam.

  References:
    Defazio et al, `The Road Less Scheduled
    <https://arxiv.org/abs/2405.15682>`_, 2024

    Defazio et al, `Schedule-Free Learning - A New Way to Train
    <https://github.com/facebookresearch/schedule_free/tree/main>`_, 2024

  Args:
    base_optimizer: Base optimizer to compute updates from.
    learning_rate: learning_rate schedule w/o decay but with warmup.
    b1: beta_1 parameter in the y update.
    weight_lr_power: we downweight the weight of averaging using this. This is
      especially helpful in early iterations during warmup.
    state_dtype: dtype for z sequence.

  Returns:
    A `GradientTransformationExtraArgs` with init and update functions.
  """
  base_optimizer = base.with_extra_args_support(base_optimizer)

  def init_fn(params: base.Params) -> ScheduleFreeState:
    z = jax.tree_util.tree_map(lambda t: t.astype(state_dtype), params)
    return ScheduleFreeState(
        b1=jnp.array([b1], dtype=jnp.float32),
        weight_sum=jnp.zeros([], dtype=jnp.float32),
        step_count=jnp.ones([], dtype=jnp.int32),
        max_lr=jnp.zeros([], dtype=jnp.float32),
        base_optimizer_state=base_optimizer.init(params),
        z=z,
    )

  def update_fn(
      grads: base.Updates,
      state: ScheduleFreeState,
      params: Optional[base.Params] = None,
      **extra_args,
  ):
    lr = learning_rate
    if callable(learning_rate):
      lr = learning_rate(state.step_count)
    max_lr = jnp.maximum(state.max_lr, lr)

    next_step_count = state.step_count + 1

    weight = max_lr**weight_lr_power
    next_total_weight = state.weight_sum + weight
    ck = weight / next_total_weight

    base_updates, next_base_optimizer_state = base_optimizer.update(
        grads,
        state.base_optimizer_state,
        params,
        **extra_args,
    )
    z = jax.tree_util.tree_map(
        lambda pi, ui: jnp.asarray(pi + ui).astype(jnp.asarray(pi).dtype),
        state.z,
        base_updates,
    )

    # Important: recompute x to both save memory and maintain accurate x seq
    # especially if y is modified by another transform wrapped on top.
    prev_x = jax.tree_util.tree_map(
        lambda yi, zi: (yi - (1.0 - b1) * zi) / b1, params, state.z
    )

    x = jax.tree_util.tree_map(
        lambda xi, zi: (1.0 - ck) * xi + ck * zi,
        prev_x,
        z,
    )
    new_params = jax.tree_util.tree_map(
        lambda xi, zi: b1 * xi + (1.0 - b1) * zi,
        x,
        z,
    )
    updates = jax.tree_util.tree_map(
        lambda npi, pi: npi - pi, new_params, params
    )

    next_state = ScheduleFreeState(
        b1=jnp.array([b1], dtype=jnp.float32),
        weight_sum=next_total_weight,
        step_count=next_step_count,
        max_lr=max_lr,
        base_optimizer_state=next_base_optimizer_state,
        z=z,
    )

    return updates, next_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
