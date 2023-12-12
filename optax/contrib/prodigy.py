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
"""Prodigy Optimizer.

A contributed implementation of the method from "Prodigy: An Expeditiously
Adaptive Parameter-Free Learner" (https://arxiv.org/abs/2306.06101) by
Konstantin Mishchenko and Aaron Defazio. A new variant of D-Adapt Adam that
adapts the learning rate faster.
"""
from typing import NamedTuple, Optional, Tuple
import chex
import jax.numpy as jnp
import jax.tree_util as tu
from optax import tree_utils
from optax._src import base
from optax._src import utils


class ProdigyState(NamedTuple):
  """State of the `GradientTransformation` returned by `prodigy`."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  # Exponential moving average of the sum of gradients.
  grad_sum: base.Updates
  # Initial point.
  params0: base.Updates
  # Distance to solution estimate.
  estim_lr: chex.Array  # shape=(), dtype=jnp.float32.
  numerator_weighted: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=int32.


def prodigy(
    learning_rate: base.ScalarOrSchedule = 0.1,
    betas: tuple[float, float] = (0.9, 0.999),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
  """Learning rate free AdamW with Prodigy.

  Implementation of the Prodigy method from "Prodigy: An Expeditiously
  Adaptive Parameter-Free Learner", a version of D-Adapt AdamW that adapts the
  baseline learning rate faster by using a weighting of the gradients that
  places higher weights on more recent gradients.
  This method works best when combined with a learning rate schedule that
  treats 1.0 as the base (usually max) value.
  References:
    [Mishchenko & Defazio, 2023](https://arxiv.org/abs/2306.06101)
  Args:
    learning_rate: Learning rate scheduling parameter. The recommended schedule
      is a linear_schedule with init_value=1.0 and end_value=0, combined with a
      0-20% learning rate warmup.
    betas: Betas for the underlying AdamW Optimizer.
    beta3: Optional momentum parameter for estimation of D.
    eps: eps for the underlying AdamW Optimizer.
    estim_lr0: Initial (under-)estimate of the learning rate.
    estim_lr_coef: LR estimates are multiplied by this parameter.
    weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain
      with add_decayed_weights.

  Returns:
    A `GradientTransformation` object.
  """
  beta1, beta2 = betas
  if beta3 is None:
    beta3 = beta2**0.5

  def init_fn(params: base.Params) -> ProdigyState:
    exp_avg = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    exp_avg_sq = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    grad_sum = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    params0 = params
    estim_lr = jnp.asarray(estim_lr0, jnp.float32)
    numerator_weighted = jnp.zeros((), jnp.float32)
    count = jnp.zeros((), jnp.int32)
    return ProdigyState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        params0,
        estim_lr,
        numerator_weighted,
        count,
    )

  def update_fn(
      updates: base.Updates,
      state: ProdigyState,
      params: Optional[base.Params] = None,
  ) -> Tuple[base.Updates, ProdigyState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    count = state.count
    sched = learning_rate(count) if callable(learning_rate) else learning_rate
    grad_sum = state.grad_sum
    params0 = state.params0
    estim_lr = state.estim_lr
    numerator_weighted = state.numerator_weighted
    bc = ((1 - beta2 ** (count + 1)) ** 0.5) / (1 - beta1 ** (count + 1))
    dlr = estim_lr * sched * bc
    dg = tu.tree_map(lambda g: estim_lr * g, updates)
    param_diff = tu.tree_map(lambda p0, p: p0 - p, params0, params)
    numerator_acum = tree_utils.tree_vdot(updates, param_diff)
    exp_avg = tu.tree_map(
        lambda ea, dgk: beta1 * ea + (1 - beta1) * dgk, state.exp_avg, dg
    )
    exp_avg_sq = tu.tree_map(
        lambda eas, dgk: beta2 * eas + (1 - beta2) * dgk * dgk,
        state.exp_avg_sq,
        dg,
    )
    grad_sum = tu.tree_map(
        lambda sk, dgk: beta3 * sk + dlr * dgk / estim_lr0, grad_sum, dg
    )
    numerator_weighted = beta3 * numerator_weighted
    numerator_weighted += (estim_lr / estim_lr0) * dlr * numerator_acum
    denominator = tree_utils.tree_sum(tu.tree_map(jnp.abs, grad_sum))
    lr_estimate = estim_lr_coef * numerator_weighted / denominator
    estim_lr = jnp.maximum(state.estim_lr, lr_estimate)
    p_update = tu.tree_map(
        lambda ea, eas, p: -weight_decay * dlr * p
        - dlr * ea / (jnp.sqrt(eas) + estim_lr * eps),
        exp_avg,
        exp_avg_sq,
        params,
    )
    new_state = ProdigyState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        params0,
        estim_lr,
        numerator_weighted,
        utils.safe_int32_increment(count),
    )
    return p_update, new_state

  return base.GradientTransformation(init_fn, update_fn)
