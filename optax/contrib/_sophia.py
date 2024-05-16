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
"""Sophia optimizer.

A contributed implementation of the Sophia optimizer from "Sophia: A Scalable
Stochastic Second-order Optimizer for Language Model Pre-training"
(https://arxiv.org/abs/2305.14342) by Hong Liu, Zhiyuan Li, David Hall,
Percy Liang, and Tengyu Ma.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform


class SophiaState(NamedTuple):
  step: jax.Array
  gradient_avg: base.Updates
  hessian: base.Updates


def scale_by_sophia(
    *,
    b1: float = 0.965,
    b2: float = 0.99,
    rho: float = 0.04,
    batch_size: int = 256,
    update_hessian_every: int = 1,
) -> base.GradientTransformation:
  """Implementation of the Sophia optimizer from arXiv:2305.14342.

  This just implements the gradient scaling step; it has to be combined with
  weight decay and learning rate scaling.

  References:
    Liu et al, 2023: https://arxiv.org/abs/2305.14342

  Args:
    b1: Exponential averaging decay constant for gradients. Must be in the range
      [0, 1).
    b2: Exponential averaging decay constant for squared gradients. Must be in
      the range [0, 1).
    rho: Scale factor for gradient updates before clipping. Must be > 0.
    batch_size: Batch size as additional scale factor for gradient updates
      before clipping. Must be > 0.
    update_hessian_every: How often to update the second order terms. As these
      are very cheap to compute, in fact (just squaring gradients), we can leave
      these at 1 by default. Must be >= 1.

  Returns:
    An optax gradient transformation for the Sophia optimizer scaling.

  Raises:
    ValueError: For invalid argument ranges.
  """

  def init_fn(params: base.Params):
    return SophiaState(
        step=jnp.array(0, dtype=jnp.int64),
        gradient_avg=jax.tree.map(jnp.zeros_like, params),
        hessian=jax.tree.map(jnp.zeros_like, params),
    )

  def update_fn(
      updates: base.Updates,
      state: SophiaState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, SophiaState]:
    del params

    # Update exponential average of gradients.
    gradient_avg = jax.tree.map(
        lambda ga, gr: ga * b1 + gr * (1 - b1),
        state.gradient_avg,
        updates,
    )

    # Update Hessian diagonal estimate, potentially every nth step.
    hessian = jax.lax.cond(
        state.step % update_hessian_every == 0,
        lambda: jax.tree.map(
            lambda he, gr: he * b2 + gr**2 * (1 - b2),
            state.hessian,
            updates,
        ),
        lambda: state.hessian,
    )

    updates = jax.tree.map(
        lambda grad_av, he: jnp.clip(
            grad_av / (rho * batch_size * he + 1e-15), -1, 1
        ),
        gradient_avg,
        hessian,
    )

    # Return new updates, and new state.
    return updates, SophiaState(
        step=state.step + 1,
        gradient_avg=gradient_avg,
        hessian=hessian,
    )

  return base.GradientTransformation(init_fn, update_fn)


def sophia(
    learning_rate: base.ScalarOrSchedule = 1e-4,
    *,
    b1: float = 0.965,
    b2: float = 0.99,
    rho: float = 0.04,
    batch_size: int = 256,
    update_hessian_every: int = 1,
    weight_decay: float = 1e-5,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Instantiates a Sophia optimizer.

  References:
    Liu et al, 2023: https://arxiv.org/abs/2305.14342

  Args:
    learning_rate: Final scaling of updates by learning rate.
    b1: Exponential averaging decay constant for gradients.
    b2: Exponential averaging decay constant for squared gradients.
    rho: Scale factor for gradient updates before clipping.
    batch_size: Batch size as additional scale factor for gradient updates
      before clipping.
    update_hessian_every: How often to update the second order terms. As these
      are very cheap to compute, in fact (just squaring gradients), we can leave
      this at 1 by default.
    weight_decay: Weight decay, applied before scaling by learning rate.
    mask: Mask for which params to apply weight decay.

  Returns:
    An optax gradient transformation for the Sophia optimizer.

  Raises:
    ValueError: For invalid argument ranges.
  """
  return combine.chain(
      # Sophia update.
      scale_by_sophia(
          b1=b1,
          b2=b2,
          rho=rho,
          batch_size=batch_size,
          update_hessian_every=update_hessian_every,
      ),
      # Weight decay will be multiplied by learning rate.
      transform.add_decayed_weights(weight_decay=weight_decay, mask=mask),
      # Scale by learning rate.
      transform.scale_by_learning_rate(learning_rate=learning_rate),
  )
