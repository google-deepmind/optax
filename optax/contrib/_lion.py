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
"""Lion Optimizer (EvoLved Sign Momentum).

Implementation of `Symbolic Discovery of Optimization Algorithms`
(https://arxiv.org/abs/2302.06675) by Chen et al.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils
import optax.tree


class ScaleByLionState(NamedTuple):
    """State for Lion Optimizer.

    Attributes:
        count: iteration count.
        mu: momentum pytree
    """

    count: chex.Array  # Shape = (), dtype = jnp.int32
    mu: base.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Rescale Updates according to Lion.

    This transforms returns updates = sign(m_tilde) where,

        m_tilde = b1 * m_{t - 1} + {1 - b1} * g_t

    and store a momentum

        m_t = b2 * m_{t - 1} + {1 - b2} * g_t

    Args:
        b1: interpolation coefficient for the signed term (paper uses ~0.9)
        b2: momentum coefficient for the Stored EMA (paper uses ~ 0.99)
        mu_dtype: Optional dtype to be used for the momentum accumulator;
                  if None dtype is inferred from params and updates.

    Returns:
        An Optax.GradientTransformation.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(updates, state: ScaleByLionState, params=None):
        del params

        # Compute m_tilde used for the sign
        m_tilde = jax.tree.map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, updates
        )

        # the returned update is sign(m_tilde)
        signed_update = jax.tree.map(jnp.sign, m_tilde)

        # Compute the stored momentum (m_t)
        new_mu = jax.tree.map(
            lambda m, g: b2 * m + (1.0 - b2) * g, state.mu, updates
        )
        new_mu = optax.tree.cast(new_mu, mu_dtype)

        new_state = ScaleByLionState(count=state.count + 1, mu=new_mu)
        return signed_update, new_state

    return base.GradientTransformation(init_fn, update_fn)


def lion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """Lion Optimizer.

    Lion uses a Sign-based updates with 2 momentum terms, making it
    memory-efficient and robust to hyperparametes choices.

    Args:
        learning_rate: A global scaling factor, either fixed or evolving
                       along iterations with a scheduler
        b1: Interpolation coefficient for the signed term
        b2: Momentum coefficient for the stored EMA
        mu_dtype: Optional dtype to be used for momentum accumulator; If
                  None then dtype is infered from params and updates
        weight_decay: Weight decay, Strength of regularization. Note that
                      this Multiplied with learning_rate
        mask: A tree with the same structure as the params PyTree,
              Or a callable that returns such a pytree given params/Updates.
              The leaves should be booleans, True for leaves/subtrees you
              want to apply the weight decay to, and False to those you
              want to skip.

    Returns:
        GradientTransformations ready to use as an optimizer
    """
    return combine.chain(
        scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )
