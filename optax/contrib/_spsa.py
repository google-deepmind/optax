# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Simultaneous Perturbation Stochastic Approximation (SPSA) method."""

from typing import Any, Callable

import jax
import optax.tree
from optax._src import base


def spsa_standard_schedule(
    init_value: float,
    decay_rate: float,
    offset: float = 0.0,
) -> optax.schedules.Schedule:
    """Returns a schedule for the SPSA learning rate or perturbation scale.

    The standard SPSA decay schedule is given by:

    .. math::
    v_k = \\frac{\\text{init\\_value}}{(count + offset)^{\\text{decay\\_rate}}}

    Args:
      init_value: The initial value of the parameter.
      decay_rate: The exponent for the polynomial decay.
      offset: The offset added to the count for stability.

    Returns:
      A function that takes the current count and returns the decayed value.
    """

    def schedule(count: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
        return init_value / ((count + offset) ** decay_rate)

    return schedule


def spsa_estimator(
    value_fn: Callable[..., jax.Array],
) -> Callable[..., base.Updates]:
    r"""Returns a function that computes the SPSA gradient estimate.

      The Simultaneous Perturbation Stochastic Approximation (SPSA) method
      estimates the gradient of a function by evaluating it at two symmetrically
      perturbed points.

      Let :math:`\Delta` be a random vector sampled from the Rademacher
    distribution
      (values in :math:`\{-1, 1\}` with equal probability). The SPSA gradient
      estimate for a function :math:`f` at point :math:`x` is given by:

      .. math::
        g = \frac{f(x + c \Delta) - f(x - c \Delta)}{2 c \Delta}

      Args:
        value_fn: The function whose gradient is to be estimated. Its first
        argument
          should be the parameters (a PyTree) with respect to which the gradient
          is estimated. It should return a scalar value.

      Returns:
        A function with the signature
        ``grad_fn(params, c, key, *args, **kwargs)``
        that computes the SPSA gradient estimate. ``c`` is the
        perturbation scale
        (a scalar). ``key`` is a ``jax.random.PRNGKey`` used to generate
      the random
        perturbations.

      References:
        Spall, `An Overview of the Simultaneous Perturbation Method
        for Efficient
        Optimization <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_,
        1998
    """

    def grad_fn(
        params: base.Params,
        c: jax.typing.ArrayLike,
        key: base.PRNGKey,
        *args: Any,
        **kwargs: Any,
    ) -> base.Updates:
        def sample_rademacher(k, shape, dtype):
            # Rademacher distribution: uniform over {-1, 1}
            return jax.random.rademacher(k, shape, dtype=dtype)

        delta = optax.tree.random_like(key, params, sampler=sample_rademacher)

        params_plus = jax.tree.map(lambda p, d: p + c * d, params, delta)
        params_minus = jax.tree.map(lambda p, d: p - c * d, params, delta)

        y_plus = value_fn(params_plus, *args, **kwargs)
        y_minus = value_fn(params_minus, *args, **kwargs)

        # Note: Since delta_i is either 1 or -1, dividing by delta_i is
        # equivalent
        # to multiplying by delta_i. We multiply for numerical stability.
        grad_estimate = jax.tree.map(
            lambda d: (y_plus - y_minus) / (2.0 * c) * d, delta
        )
        return grad_estimate

    return grad_fn
