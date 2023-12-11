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
"""D-Adatation (AdamW variant).

A contributed implementation of the method from "Learning-Rate-Free Learning by
D-Adaptation" (https://arxiv.org/abs/2301.07733) by Aaron Defazio and Konstantin
Mishchenko (ICML 2023 Outstanding Paper award).
"""
from typing import NamedTuple, Optional, Tuple
import chex
import jax.numpy as jnp
import jax.tree_util as tu
from optax import tree_utils
from optax._src import base
from optax._src import utils


class DAdaptAdamWState(NamedTuple):
  """State of the `GradientTransformation` returned by `dadapt_adamw`."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  # Exponential moving average of the sum of gradients.
  grad_sum: base.Updates  # shape=(), dtype=jnp.float32.
  # Distance to solution estimate.
  estim_lr: chex.Array  # shape=(), dtype=jnp.float32.
  numerator_weighted: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=jnp.int32.


def dadapt_adamw(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    weight_decay: float = 0.,
) -> base.GradientTransformation:
  """Learning rate free AdamW by D-Adaptation.

  Adapts the baseline learning rate of AdamW automatically by estimating the
  initial distance to solution in the infinity norm.
  This method works best when combined with a learning rate schedule that
  treats 1.0 as the base (usually max) value.
  References:
    [Defazio & Mishchenko, 2023](https://arxiv.org/abs/2301.07733)
  Args:
    learning_rate: Learning rate scheduling parameter. The recommended schedule
      is a linear_schedule with init_value=1.0 and end_value=0, combined with a
      0-20% learning rate warmup.
    betas: Betas for the underlying AdamW Optimizer.
    eps: eps for the underlying AdamW Optimizer.
    estim_lr0: Initial (under-)estimate of the learning rate.
    weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain
      with add_decayed_weights.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params: base.Params) -> DAdaptAdamWState:
    exp_avg = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    exp_avg_sq = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    grad_sum = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    estim_lr = jnp.asarray(estim_lr0, jnp.float32)
    numerator_weighted = jnp.zeros([], jnp.float32)
    count = jnp.zeros([], jnp.int32)
    return DAdaptAdamWState(
        exp_avg, exp_avg_sq, grad_sum, estim_lr, numerator_weighted, count
    )

  def update_fn(
      updates: base.Updates,
      state: DAdaptAdamWState,
      params: Optional[base.Params] = None,
  ) -> Tuple[base.Updates, DAdaptAdamWState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    count = state.count
    beta1, beta2 = betas
    sb2 = beta2 ** (0.5)
    sched = learning_rate(count) if callable(learning_rate) else learning_rate
    grad_sum = state.grad_sum
    numerator_weighted = state.numerator_weighted
    bc = ((1 - beta2 ** (count + 1)) ** 0.5) / (1 - beta1 ** (count + 1))
    dlr = state.estim_lr * sched * bc
    s_weighted = tu.tree_map(
        lambda sk, eas: sk / (jnp.sqrt(eas) + eps), grad_sum, state.exp_avg_sq
    )
    numerator_acum = tree_utils.tree_vdot(updates, s_weighted)
    exp_avg = tu.tree_map(
        lambda ea, g: beta1 * ea + (1 - beta1) * dlr * g, state.exp_avg, updates
    )
    exp_avg_sq = tu.tree_map(
        lambda eas, g: beta2 * eas + (1 - beta2) * g * g,
        state.exp_avg_sq,
        updates,
    )
    grad_sum = tu.tree_map(
        lambda sk, g: sb2 * sk + (1 - sb2) * dlr * g, grad_sum, updates
    )
    grad_sum_l1 = tree_utils.tree_sum(tu.tree_map(jnp.abs, grad_sum))
    numerator_weighted = (
        sb2 * numerator_weighted + (1 - sb2) * dlr * numerator_acum
    )
    d_estimate = numerator_weighted / ((1 - sb2) * grad_sum_l1)
    estim_lr = jnp.maximum(state.estim_lr, d_estimate)
    p_update = tu.tree_map(
        lambda ea, eas, p: -weight_decay * dlr * p - ea / (jnp.sqrt(eas) + eps),
        exp_avg,
        exp_avg_sq,
        params,
    )
    new_state = DAdaptAdamWState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        estim_lr,
        numerator_weighted,
        utils.safe_int32_increment(count),
    )
    return p_update, new_state

  return base.GradientTransformation(init_fn, update_fn)
