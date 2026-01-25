# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""MADGRAD optimizer."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax import tree_utils


class MadgradState(NamedTuple):
  """State for the MADGRAD optimizer."""
  count: jax.Array
  grad_sum_sq: base.Updates  # Weighted sum of squared gradients (G_k)
  s: base.Updates            # Weighted sum of gradients (s_k)
  x0: base.Params            # Initial parameters (x_0)


def scale_by_madgrad(
    learning_rate: base.ScalarOrSchedule,
    momentum: float = 0.9,
    eps: float = 1e-6,
) -> base.GradientTransformation:
  """Rescale updates according to the MADGRAD algorithm.

  MADGRAD is a Dual Averaging method that maintains a weighted sum of gradients
  and squared gradients to compute adaptive updates.

  References:
    Defazio et al, `Adaptivity without Compromise: A Momentumized, Adaptive,
    Dual Averaged Gradient Method for Stochastic Optimization
    <https://arxiv.org/abs/2101.11075>`_, 2021

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    momentum: Momentum parameter.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    return MadgradState(
        count=jnp.zeros([], jnp.int32),
        grad_sum_sq=tree_utils.tree_zeros_like(params),
        s=tree_utils.tree_zeros_like(params),
        x0=tree_utils.tree_cast(params, params.dtype),
    )

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    count = state.count
    if callable(learning_rate):
      lr = learning_rate(count)
    else:
      lr = learning_rate

    # Ensure stability by adding eps to the learning rate, matching the
    # official PyTorch implementation.
    lr_stable = lr + eps

    # lamb = lr * sqrt(k + 1)
    lamb = lr_stable * jnp.sqrt(count + 1)

    # G_{k+1} = G_k + lamb * g_k^2
    grad_sum_sq = jax.tree.map(
        lambda g_sq, g: g_sq + lamb * (g ** 2), state.grad_sum_sq, updates
    )

    # s_{k+1} = s_k + lamb * g_k
    s = jax.tree.map(
        lambda s_val, g: s_val + lamb * g, state.s, updates
    )

    # sigma_{k+1} = (G_{k+1})^(1/3) + eps
    sigma = jax.tree.map(
        lambda g_sq: jnp.cbrt(g_sq) + eps, grad_sum_sq
    )

    # z_{k+1} = x_0 - s_{k+1} / sigma_{k+1}
    z = jax.tree.map(
        lambda x0_val, s_val, sig: x0_val - s_val / sig, state.x0, s, sigma
    )

    # x_{k+1} = (1 - momentum) * z_{k+1} + momentum * x_k
    x_new = jax.tree.map(
        lambda z_val, x_k: (1 - momentum) * z_val + momentum * x_k,
        z,
        params
    )

    # Convert the new parameter state into an update (x_new - x_old)
    final_updates = jax.tree.map(lambda n, o: n - o, x_new, params)

    new_state = MadgradState(
        count=numerics.safe_increment(count),
        grad_sum_sq=grad_sum_sq,
        s=s,
        x0=state.x0,
    )
    return final_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def madgrad(
    learning_rate: base.ScalarOrSchedule,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    eps: float = 1e-6,
) -> base.GradientTransformation:
  """The MADGRAD optimizer.

  MADGRAD is a general purpose optimizer that matches the performance of
  SGD+Momentum on vision tasks and Adam on NLP tasks.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    momentum: Momentum parameter (default: 0.9).
    weight_decay: Strength of the weight decay regularization (L2).
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay),
      scale_by_madgrad(learning_rate=learning_rate, momentum=momentum, eps=eps)
  )
