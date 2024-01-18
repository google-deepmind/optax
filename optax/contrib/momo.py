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

A contributed implementation of the method from "" (https://arxiv.org/abs/2305.07583) by Fabian Schaipp, 
Ruben Ohana, Michael Eickenberg, Aaron Defazio and Robert M. Gower.
"""
from typing import NamedTuple, Optional
import jax.numpy as jnp
import jax.tree_util as tu
from jax import Array
from optax import tree_utils
from optax._src import base
from optax._src import utils

class MomoState(NamedTuple):
  """State of the `GradientTransformation` returned by `momo`."""
  exp_avg: base.Updates       # dk 
  barf: float                 # bar f_k
  gamma: float                # gamma_k
  count: float                # iteration counter


def momo(
    learning_rate: base.ScalarOrSchedule = 1.0,
    beta: float = 0.9,
    lb: float = 0.0,
    weight_decay: float = 0,
    delta: float = 1e-10
) -> base.GradientTransformationExtraArgs:
  """Adaptive Learning Rates for SGD with momentum.

  MoMo typically needs less tuning for value of `learning_rate`,
  by exploting the fact that a lower bound of the loss (or the optimal value) is known.
  For most tasks, zero is a lower bound and an accurate estimate of the final loss.

  MoMo performs SGD with momentum with a Polyak-type learning rate. The effective step size is
    `min(learning_rate, <adaptive term>)`

  where the adaptive term is computed on the fly. 

  Note that in `update_fn` you need to pass the latest (batch) loss to the argument `loss`.

  References:
    [Schaipp et al., 2023](https://arxiv.org/abs/2305.07583)
  Args:
    learning_rate: User-specified learning rate. Recommended to be chosen rather large, by default 1.0.
    betas: Momentum coefficient (for EMA).
    lb: Lower bound of the loss. Zero should be a good choice for many tasks.
    weight_decay: Weight-decay parameter.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params: base.Params) -> MomoState:
    exp_avg = tu.tree_map(lambda p: jnp.zeros(p.shape), params)
    barf = 0
    gamma = 0
    count = 0
    
    return MomoState(exp_avg, barf, gamma, count)

  def update_fn(
      updates: base.Updates,
      state: MomoState,
      params: Optional[base.Params],
      *,
      loss: Optional[Array]) -> tuple[base.Updates, MomoState]:
      # update: latest gradient g_k
      # loss: latest loss f(xk,sk)
      # params: latest xk
    
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    
    count = state.count
    if count == 0: # initialize at first gradient, and loss 
      barf = loss
      exp_avg = tu.tree_map(lambda g: g, updates)                               # clone gradients
      gamma = tree_utils.tree_vdot(updates, params)     
    else:
      barf = beta*state.barf + (1-beta)*loss
      exp_avg = tu.tree_map(lambda ea, g: beta*ea + (1-beta)*g,
                          state.exp_avg, updates)                               # dk
      gamma = beta*state.gamma + (1-beta)*tree_utils.tree_vdot(updates, params)     
    

    exp_avg_norm = tree_utils.tree_l2_norm(exp_avg, squared=True)               # should do the same

    iprod = tree_utils.tree_vdot(exp_avg, params)                               # <dk,xk>
    t1 = jnp.maximum(barf + iprod - gamma - lb, 0)/(exp_avg_norm+delta)
    
    alpha = learning_rate(count) if callable(learning_rate) else learning_rate
    tau = jnp.minimum(alpha, t1)
    
    p_update = tu.tree_map(lambda ea, p:
                        -(alpha*weight_decay)/(1+alpha*weight_decay)*p
                        - tau*ea,
                        exp_avg, params
                        )
    
    new_state = MomoState(exp_avg=exp_avg, 
                          barf=barf, 
                          gamma=gamma, 
                          count=utils.safe_int32_increment(count))
    
    return p_update, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)