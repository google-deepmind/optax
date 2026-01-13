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
"""MARS optimizer implementation."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils

__all__ = ["mars_adamw", "scale_by_mars", "MarsState"]


class MarsState(NamedTuple):
    """State for the MARS algorithm."""

    count: chex.Array
    mu: base.Updates
    nu: base.Updates
    prev_grad: base.Updates


def scale_by_mars(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    gamma: float = 0.025,
    max_norm: Optional[float] = None,
) -> base.GradientTransformation:
    """Rescale updates according to the MARS algorithm.

    MARS (Make vAriance Reduction Shine) consists of a standard prior
    correction term (like Adam) plus a correction term based on the difference
    between the current and previous gradients.

    References:
      Hu et al., 2024: https://arxiv.org/abs/2411.10438

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      gamma: Coefficient for the correction term.
      max_norm: Optional maximum norm for the correction term. If None, no
        clipping is applied to the correction.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        return MarsState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree_util.tree_map(jnp.zeros_like, params),
            nu=jax.tree_util.tree_map(jnp.zeros_like, params),
            prev_grad=jax.tree_util.tree_map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        del params
        # At t=0, state.count is 0.
        # We want to mask the correction at t=0.
        is_step_zero = state.count == 0

        count = state.count + jnp.array(1, dtype=jnp.int32)

        mu = tree.update_moment(updates, state.mu, b1, 1)
        nu = tree.update_moment(updates, state.nu, b2, 2)

        mu_hat = tree.bias_correction(mu, b1, count)
        nu_hat = tree.bias_correction(nu, b2, count)

        adam_direction = jax.tree_util.tree_map(
            lambda m, n: m / (jnp.sqrt(n) + eps), mu_hat, nu_hat
        )

        # Correction Calculation
        # coeff = gamma * (b1 / (1.0 - b1))
        correction_coeff = gamma * (b1 / (1.0 - b1))

        # raw_correction = coeff * (g_t - prev_grad)
        raw_correction = jax.tree_util.tree_map(
            lambda g, pg: correction_coeff * (g - pg), updates, state.prev_grad
        )

        # Clipping the correction term
        correction = raw_correction
        if max_norm is not None:
            global_norm = numerics.safe_norm(correction, min_norm=0.0)
            scale = jnp.where(
                global_norm > max_norm,
                max_norm / (global_norm + 1e-12),  # Avoid division by zero
                1.0,
            )
            correction = jax.tree_util.tree_map(lambda t: t * scale, correction)

        # Apply mask for t=0. If step 0, correction is 0.
        correction = jax.tree_util.tree_map(
            lambda c: jnp.where(is_step_zero, jnp.zeros_like(c), c), correction
        )

        new_updates = jax.tree_util.tree_map(
            lambda adam, corr: adam + corr, adam_direction, correction
        )

        new_state = MarsState(
            count=count,
            mu=mu,
            nu=nu,
            prev_grad=updates,
        )

        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def mars_adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    gamma: float = 0.025,
    max_norm: Optional[float] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """A MARS optimizer with AdamW features.

    Args:
      learning_rate: A scalar or a schedule for the learning rate.
      b1: The exponential decay rate for the 1st moment estimates.
      b2: The exponential decay rate for the 2nd moment estimates.
      eps: A small constant for numerical stability.
      weight_decay: Strength of the weight decay regularization.
      gamma: Coefficient for the MARS correction term.
      max_norm: Maximum norm for the correction term clipping.
      mask: A tree with same structure as (or prefix of) params containing
        booleans, which indicate whether to apply the weight decay to the
        corresponding parameter.

    Returns:
      A `GradientTransformation` representing the MARS AdamW optimizer.
    """
    return combine.chain(
        scale_by_mars(
            b1=b1,
            b2=b2,
            eps=eps,
            gamma=gamma,
            max_norm=max_norm,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )
