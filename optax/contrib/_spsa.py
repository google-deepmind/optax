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
"""SPSA: Simultaneous Perturbation Stochastic Approximation.

A contributed implementation of the gradient-free SPSA method from
"Multivariate Stochastic Approximation Using a Simultaneous Perturbation
Gradient Approximation" (https://doi.org/10.1109/9.119632) by James C. Spall.

The objective is supplied at update time through the ``obj_fn`` keyword
argument, following the same convention as
:func:`optax.contrib.hutchinson_estimator_diag_hessian`.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
import optax.tree


class SPSAState(NamedTuple):
  """State for the SPSA gradient estimator."""

  count: jax.Array  # shape=(), dtype=jnp.int32
  key: jax.Array


def spsa_gradient(
    c: jax.typing.ArrayLike = 0.1,
    gamma: jax.typing.ArrayLike = 0.101,
    seed: Optional[jax.Array] = None,
) -> base.GradientTransformationExtraArgs:
  r"""Estimates the gradient via Simultaneous Perturbation Stochastic Approx.

  SPSA is a gradient-free method: rather than differentiating the objective, it
  estimates the whole gradient from only **two** objective evaluations per step,
  regardless of the number of parameters. This is useful when the objective is
  non-differentiable, only available as a black box, or expensive to
  differentiate.

  At step :math:`k` a random perturbation vector :math:`\Delta_k` with i.i.d.
  :math:`\pm 1` entries is drawn and the gradient estimate is

  .. math::
    \hat{g}_{k,i} = \frac{f(\theta + c_k \Delta_k) - f(\theta - c_k \Delta_k)}
                         {2\, c_k\, \Delta_{k,i}},
    \qquad c_k = \frac{c}{(k + 1)^\gamma}.

  Because :math:`\Delta_{k,i} \in \{-1, +1\}` we have
  :math:`1 / \Delta_{k,i} = \Delta_{k,i}`, so the estimate is a scalar times
  :math:`\Delta_k`. Its expectation matches the true gradient up to
  :math:`O(c_k^2)`. The returned updates are this gradient estimate; compose it
  with a step size (e.g. :func:`optax.scale_by_learning_rate`) or use the
  :func:`optax.contrib.spsa` wrapper.

  The objective is supplied at update time via the ``obj_fn`` keyword argument,
  exactly like :func:`optax.contrib.hutchinson_estimator_diag_hessian`.
  ``obj_fn`` must take ``params`` as its only argument and return a scalar::

    obj_fn = lambda params: loss_fn(params, batch)
    grad_estimate, state = estimator.update(grads, state, params, obj_fn=obj_fn)

  Args:
    c: Base perturbation size :math:`c`; must be positive.
    gamma: Decay exponent for the perturbation size. Spall recommends ``0.101``.
    seed: Optional PRNG key used to draw the perturbation vectors.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  References:
    Spall, `Multivariate Stochastic Approximation Using a Simultaneous
    Perturbation Gradient Approximation
    <https://doi.org/10.1109/9.119632>`_, IEEE TAC, 1992.

  .. seealso:: :func:`optax.contrib.spsa`
  """

  def init_fn(params):
    del params
    key = seed if seed is not None else jax.random.PRNGKey(0)
    return SPSAState(count=jnp.zeros([], jnp.int32), key=key)

  def update_fn(updates, state, params=None, obj_fn=None, **extra_args):
    # Complies with the GradientTransformationExtraArgs signature but ignores
    # the incoming ``updates`` (SPSA estimates its own gradient) and any other
    # extra args.
    del updates, extra_args
    if params is None:
      raise ValueError('params must be provided to the spsa update function.')
    if obj_fn is None:
      raise ValueError('obj_fn must be provided to the spsa update function.')

    key, subkey = jax.random.split(state.key)
    perturbation = optax.tree.random_like(
        subkey, params, jax.random.rademacher, dtype=jnp.float32
    )
    perturbation = optax.tree.cast(
        perturbation, optax.tree.dtype(params, 'lowest')
    )

    step = jnp.asarray(state.count, jnp.float32) + 1.0
    ck = c / step**gamma

    params_plus = jax.tree.map(lambda p, d: p + ck * d, params, perturbation)
    params_minus = jax.tree.map(lambda p, d: p - ck * d, params, perturbation)
    delta_obj = (obj_fn(params_plus) - obj_fn(params_minus)) / (2.0 * ck)

    # 1 / Delta_i == Delta_i for Delta_i in {-1, +1}.
    grad_estimate = jax.tree.map(lambda d: delta_obj * d, perturbation)
    return grad_estimate, SPSAState(
        count=numerics.safe_increment(state.count), key=key
    )

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def spsa(
    learning_rate: base.ScalarOrSchedule,
    c: jax.typing.ArrayLike = 0.1,
    gamma: jax.typing.ArrayLike = 0.101,
    seed: Optional[jax.Array] = None,
) -> base.GradientTransformationExtraArgs:
  r"""The SPSA (gradient-free) optimizer.

  Combines the SPSA gradient estimate (:func:`optax.contrib.spsa_gradient`) with
  a learning rate. Only objective *values* are used, so the objective need not
  be differentiable. The objective is passed at update time through ``obj_fn``
  (a function of ``params`` returning a scalar); the incoming ``grads`` are
  ignored and may be ``None``-like::

    obj_fn = lambda params: loss_fn(params, batch)
    opt = optax.contrib.spsa(learning_rate=0.1)
    state = opt.init(params)
    updates, state = opt.update(grads, state, params, obj_fn=obj_fn)
    params = optax.apply_updates(params, updates)

  Args:
    learning_rate: The step size :math:`a_k`, either fixed or a schedule. Spall's
      classic decaying choice ``a / (A + k + 1) ** alpha`` can be built with
      :func:`optax.polynomial_schedule`.
    c: Base perturbation size; must be positive.
    gamma: Decay exponent for the perturbation size.
    seed: Optional PRNG key used to draw the perturbation vectors.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  References:
    Spall, `Multivariate Stochastic Approximation Using a Simultaneous
    Perturbation Gradient Approximation
    <https://doi.org/10.1109/9.119632>`_, IEEE TAC, 1992.

  .. seealso:: :func:`optax.contrib.spsa_gradient`
  """
  return combine.chain(
      spsa_gradient(c=c, gamma=gamma, seed=seed),
      transform.scale_by_learning_rate(learning_rate),
  )
