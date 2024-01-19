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
"""MoMo.
A contributed implementation of the method from 
"MoMo: Momentum Models for Adaptive Learning Rates" 
(https://arxiv.org/abs/2305.07583) by Fabian Schaipp, Ruben Ohana,
Michael Eickenberg, Aaron Defazio and Robert M. Gower.
"""
from typing import NamedTuple, Optional
import chex
import jax.numpy as jnp
import jax.tree_util as tu
from jax import Array
from jax.lax import cond
from optax import tree_utils
from optax._src import base
from optax._src import utils

class MomoState(NamedTuple):
  """State of the `GradientTransformation` returned by `momo`."""
  exp_avg: base.Updates
  barf: float
  gamma: float
  count: chex.Array

def momo(
    learning_rate: base.ScalarOrSchedule = 1.0,
    beta: float = 0.9,
    lb: float = 0.0,
    weight_decay: float = 0,
    delta: float = 1e-10
) -> base.GradientTransformationExtraArgs:
  """Adaptive Learning Rates for SGD with momentum.

  MoMo typically needs less tuning for value of `learning_rate`,
  by exploting the fact that a lower bound of the loss (or the optimal value) is
  known. For most tasks, zero is a lower bound and an accurate estimate of the
  final loss.

  MoMo performs SGD with momentum with a Polyak-type learning rate. The 
  effective step size is
    `min(learning_rate, <adaptive term>)`

  where the adaptive term is computed on the fly. 

  Note that in `update_fn` you need to pass the latest (batch) loss to
    the argument `loss`.

  References:
    [Schaipp et al., 2023](https://arxiv.org/abs/2305.07583)
  Args:
    learning_rate: User-specified learning rate. Recommended to be chosen
      rather large, by default 1.0.
    beta: Momentum coefficient (for EMA).
    lb: Lower bound of the loss. Zero should be a good choice for many tasks.
    weight_decay: Weight-decay parameter.

  Returns:
    A `GradientTransformation` object.
  """
  def init_fn(params: base.Params) -> MomoState:
    exp_avg = tu.tree_map(lambda p: jnp.zeros(p.shape), params)
    barf = 0
    gamma = 0
    count = jnp.zeros([], jnp.int32)
    return MomoState(exp_avg, barf, gamma, count)

  def update_fn(
      updates: base.Updates,
      state: MomoState,
      params: Optional[base.Params],
      loss: Optional[Array] = None) -> tuple[base.Updates, MomoState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    if loss is None:
      raise ValueError("""You need to pass the latest loss value to Momo.
                       Use `jax.value_and_grad` for this.""")
    count = state.count
    # initialize at first gradient, and loss
    bt = cond(count == 0, lambda: 0., lambda: beta)
    barf = bt*state.barf + (1-bt)*loss
    exp_avg = tu.tree_map(
      lambda ea, g: bt*ea + (1-bt)*g,
      state.exp_avg,
      updates
      )
    gamma = bt*state.gamma + (1-bt)*tree_utils.tree_vdot(updates, params)
    exp_avg_norm = tree_utils.tree_l2_norm(exp_avg,squared=True)
    iprod = tree_utils.tree_vdot(exp_avg, params)
    alpha = learning_rate(count) if callable(learning_rate) else learning_rate
    t1 = jnp.maximum((1+alpha*weight_decay)*(
                            barf - lb - gamma
                            )  + iprod , 0)/(exp_avg_norm+delta)
    tau = jnp.minimum(alpha, t1)
    p_update = tu.tree_map(
      lambda ea, p:
      -(alpha*weight_decay)/(1+alpha*weight_decay)*p
      - tau*ea,
      exp_avg, params
    )
    new_state = MomoState(
      exp_avg=exp_avg,
      barf=barf,
      gamma=gamma,
      count=utils.safe_int32_increment(count)
    )
    return p_update, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
