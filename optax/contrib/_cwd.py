# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Cautious Weight Decay."""

from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src import wrappers
from optax.transforms._adding import WeightDecaySchedule


def add_cautious_weight_decay(
    weight_decay: base.ScalarOrSchedule = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Add cautious weight decay.

  Performs weight decay only on parameters where the sign of the parameter
  and the sign of the update are the same.

  References:
    Chen et al., "Cautious Weight Decay", 2025. https://arxiv.org/abs/2510.12402

  Args:
    weight_decay: A scalar weight decay rate or a schedule.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    del params
    if callable(weight_decay):
      return WeightDecaySchedule(count=jnp.zeros([], jnp.int32))
    else:
      return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    if callable(weight_decay):
      new_state = WeightDecaySchedule(
          count=numerics.safe_increment(state.count)
      )
    else:
      new_state = state

    # If weight decay is a zero constant, we can skip the update.
    if isinstance(weight_decay, (int, float)) and weight_decay == 0.0:
      return updates, new_state

    s = weight_decay(state.count) if callable(weight_decay) else weight_decay

    def _cwd_update(u, p):
      if u is None:
        return None
      # Cautious Weight Decay: only decay if signs align (u * p >= 0)
      # We use sign(u) * sign(p) >= 0 to avoid potential overflow in u * p
      mask_align = jnp.sign(u) * jnp.sign(p) >= 0
      return u + s * mask_align * p

    updates = jax.tree.map(
        _cwd_update,
        updates,
        params,
        is_leaf=lambda x: x is None,
    )
    return updates, new_state

  # If mask is not `None`, apply mask to the gradient transformation.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask
    )
  return base.GradientTransformation(init_fn, update_fn)
