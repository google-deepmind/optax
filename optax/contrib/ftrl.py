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
"""FTRL Optimizer.

A contributed implemention of the optimization algorithm "Follow The Regularized
Leader" (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
by McMahan et al. (2013).
"""
from typing import NamedTuple, Optional, Tuple
import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from optax import tree_utils as tu
from optax._src import base
from optax._src import utils


class FtrlState(NamedTuple):
    """State of the `GradientTransformation` returned by `ftrl`."""
    z: base.Updates
    n: base.Updates
    
    
def ftrl(
  learning_rate: float = 0.001,
  learning_rate_power: float = 0.5,
  lambda_1: float = 0,
  lambda_2: float = 0,
  beta: float = 0,
) -> base.GradientTransformation:
  """
  
  Implementation of the FTRL optimization algorithm from "Follow The Regularized Leader"
  
  References:
    McMahan et al, `Ad Click Prediction: a View from the Trenches
    <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf>`_, 2013
    [Keras implementation](https://keras.io/api/optimizers/ftrl)
    
  Args:
    learning_rate: learning rate (same as alpha in the paper)
    learning_rate_power: Controls how the learning rate decreases during
      training. Use zero for a fixed learning rate.
    # initial_accumulator_value: The starting value for accumulators.
    lambda_1: l1 regularization strength
    lambda_2: l2 regularization strength
    beta: same as beta in the paper
  
  Returns:
    A `GradientTransformation` object.
  """
  
  def init_fn(params: base.Params) -> FtrlState:
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    n = jax.tree_util.tree_map(jnp.zeros_like, params)
    return FtrlState(z=z, n=n)
    
  def update_fn(updates: base.Updates, 
                state: FtrlState, 
                params: Optional[base.Params] = None,
                ) -> Tuple[base.Updates, FtrlState]:
    
    # TODO: add checks for dtypes and values
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    
    z = state.z
    n = state.n
    alpha = learning_rate
    
    # find w_t
    mask = jax.tree_util.tree_map(lambda x: abs(x) >= lambda_1, z).astype(float)
    sgn_z = jax.tree_util.tree_map(jnp.sign, z)
    numerator = tu.tree_scalar_mul(lambda_1, sgn_z)
    numerator = tu.tree_sub(numerator, z)
    
    root_n = jax.tree_util.tree_map(jnp.sqrt, n)
    denominator = jax.tree_util.tree_map(lambda x: (x + beta) / alpha + lambda_2, root_n)
    prev_w = tu.tree_div(numerator, denominator)
    prev_w = tu.tree_mul(prev_w, mask)
    
    # update z, n
    g_squared = jax.tree_util.tree_map(jnp.square, updates)
    inside = tu.tree_add(n, g_squared)
    root_inside = jax.tree_util.tree_map(jnp.sqrt, inside)
    sigma = tu.tree_sub(root_inside, root_n)
    sigma = tu.tree_scalar_mul(1 / alpha, sigma)
    
    z = tu.tree_sub(tu.tree_add(z, updates), tu.tree_mul(sigma, prev_w))
    n = inside
    
    # find w_t+1
    mask = jax.tree_util.tree_map(lambda x: abs(x) >= lambda_1, z).astype(float)
    sgn_z = jax.tree_util.tree_map(jnp.sign, z)
    numerator = tu.tree_scalar_mul(lambda_1, sgn_z)
    numerator = tu.tree_sub(numerator, z)
    
    root_n = jax.tree_util.tree_map(jnp.sqrt, n)
    denominator = jax.tree_util.tree_map(lambda x: (x + beta) / alpha + lambda_2, root_n)
    w = tu.tree_div(numerator, denominator)
    w = tu.tree_mul(w, mask)
    
    return w - prev_w, FtrlState(z=z, n=n)
    
  return base.GradientTransformation(init_fn, update_fn)