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
"""RLion: A Refined Lion Optimizer.

Implementation of `RLion: A Refined Lion Optimizer Implementation`
(https://www.researchgate.net/publication/385679808_RLion_A_Refined_Lion_Optimizer_for_Deep_Learning).
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


class ScaleByRLionState(NamedTuple):
    """State for RLion Optimizer.

    Attributes:
        count: Iteration count.
        mu: momentum pytree
    """
    count: chex.Array
    mu: base.Updates


def smooth_sign(x: chex.Array, beta: float = 1.0) -> chex.Array:
    """Smooth approximation of the sign function using tanh.

    Args:
        x: input array
        beta: smoothness parameter

    Returns:
        Smooth sign of x.
    """
    return jnp.tanh(beta * x)


def scale_by_rlion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    use_smooth_sign: bool = True,
    smooth_beta: float = 1.0,
) -> base.GradientTransformation:
    """Scale updates by RLion algorithm.

    Args:
        b1: decay rate for the 1st moment estimates.
        b2: decay rate for the 2nd moment estimates.
        mu_dtype: data type for the momentum.
        use_smooth_sign: whether to use smooth sign function.
        smooth_beta: smoothness parameter for smooth sign function.

    Returns:
        A GradientTransformation.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params: base.Params) -> ScaleByRLionState:
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        return ScaleByRLionState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(
        updates: base.Updates,
        state: ScaleByRLionState,
        params: Optional[base.Params] = None,
    ) -> tuple[base.Updates, ScaleByRLionState]:
        del params

        # Compute m_tilde for sign
        m_tilde = jax.tree.map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, updates
        )

        # Apply sign function
        if use_smooth_sign:
            signed_updates = jax.tree.map(
                lambda x: smooth_sign(x, smooth_beta), m_tilde
            )
        else:
            signed_updates = jax.tree.map(jnp.sign, m_tilde)

        # Update momentum
        new_mu = jax.tree.map(
            lambda m, g: b2 * m + (1.0 - b2) * g, state.mu, updates
        )
        new_mu = optax.tree.cast(new_mu, mu_dtype)

        new_state = ScaleByRLionState(count=state.count + 1, mu=new_mu)
        return signed_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def rlion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    use_smooth_sign: bool = True,
    smooth_beta: float = 1.0,
) -> base.GradientTransformation:
    """RLion optimizer with smooth sign approximation.

    RLion improves upon the standard Lion optimizer by using a smooth
    sign approximation (tanh) instead of the hard sign function, providing
    better gradient flow and numerical stability, especially in FP16 training.

    Args:
        learning_rate: Learning rate (scalar or schedule).
        b1: Decay rate for the 1st moment estimates.
        b2: Decay rate for the 2nd moment estimates.
        mu_dtype: Optional dtype for momentum accumulator.
        weight_decay: Weight decay coefficient.
        mask: Optional mask for weight decay application.
        use_smooth_sign: Whether to use smooth sign (tanh) or hard sign.
        smooth_beta: Smoothness parameter for tanh approximation.

    Returns:
        A GradientTransformation.

    Examples:
        >>> import optax
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def f(x): return jnp.sum(x ** 2)
        >>> solver = optax.contrib.rlion(learning_rate=0.001)
        >>> params = jnp.array([1., 2., 3.])
        >>> opt_state = solver.init(params)
        >>> for _ in range(5):
        ...     grad = jax.grad(f)(params)
        ...     updates, opt_state = solver.update(grad, opt_state, params)
        ...     params = optax.apply_updates(params, updates)

    References:
        RLion: A Refined Lion Optimizer Implementation
        https://www.researchgate.net/publication/385679808
    """
    return combine.chain(
        scale_by_rlion(
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype,
            use_smooth_sign=use_smooth_sign,
            smooth_beta=smooth_beta,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )
