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
from optax import tree_utils
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
    [McMahan et al., 2013](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
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
    
    return
    
  return base.GradientTransformation(init_fn, update_fn)