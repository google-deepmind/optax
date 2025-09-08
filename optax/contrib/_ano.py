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
"""ANO (Ano: Faster is Better in Noisy Landscapes)."""

from typing import Any, Optional, Callable
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


def scale_by_ano(
    b1: float = 0.92,
    b2: float = 0.99,
    eps: float = 1e-8,
    logarithmic_schedule: bool = False,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the ANO algorithm.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate parameter used in the sign-based second-moment update.
    eps: Term added to the denominator to improve numerical stability.
    logarithmic_schedule: If True, use logarithmic
      schedule for b1: 1-1/log(max(2,k)).
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. seealso:: :func:`optax.contrib.ano`
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment m_0
    nu = optax.tree.zeros_like(params)  # Second moment v_0
    return transform.ScaleByAdamState(
        count=jnp.zeros([], jnp.int32), mu=mu, nu=nu
    )

  def update_fn(updates, state, params=None):
    del params
    g = updates
    count_inc = numerics.safe_increment(state.count)

    # Compute scalar b1 schedule (float32 host scalar), then cast per-leaf.
    if logarithmic_schedule:
      step = count_inc.astype(jnp.float32)
      max_step = jnp.maximum(jnp.asarray(2.0, dtype=step.dtype), step)
      b1_dynamic_scalar = 1.0 - 1.0 / jnp.log(max_step)
    else:
      b1_dynamic_scalar = jnp.asarray(b1, dtype=jnp.float32)

    # First moment: m_t = b1 * m_{t-1} + (1 - b1) * g_t
    # Cast b1 per-leaf to avoid promotion.
    def _update_mu(g_t, m_prev):
      b1_t = jnp.asarray(b1_dynamic_scalar, dtype=m_prev.dtype)
      one = jnp.asarray(1.0, dtype=m_prev.dtype)
      return b1_t * m_prev + (one - b1_t) * g_t

    mu = jax.tree.map(_update_mu, g, state.mu)

    # Second moment with sign-based EMA (formula preserved):
    # v_t = b2 * v_{t-1} + (1 - b2) * sign(g_t^2 - v_{t-1}) * g_t^2
    # Cast b2 and (1-b2) per-leaf to avoid promotion.
    def _update_v(g_t, v_prev):
      g2 = jnp.square(g_t).astype(v_prev.dtype)
      b2_t = jnp.asarray(b2, dtype=v_prev.dtype)
      one_minus_b2_t = jnp.asarray(1.0 - b2, dtype=v_prev.dtype)
      sign_term = jnp.sign(g2 - v_prev)
      return b2_t * v_prev + one_minus_b2_t * sign_term * g2

    nu = jax.tree.map(_update_v, g, state.nu)

    # Bias correction for second moment (scalar), cast per-leaf at use-site.
    bias_correction2_scalar = (
      1.0 - jnp.asarray(b2, dtype=jnp.float32) ** count_inc
    )

    # Direction: |g| * sign(m) / sqrt(v_hat + eps), all in leaf dtype.
    def _direction(g_t, m_t, v_t):
      bc2 = jnp.asarray(bias_correction2_scalar, dtype=v_t.dtype)
      v_hat = v_t / bc2
      eps_t = jnp.asarray(eps, dtype=v_t.dtype)
      denom = jnp.sqrt(v_hat + eps_t)
      sgn = jnp.sign(m_t).astype(g_t.dtype)
      return jnp.abs(g_t) * sgn / denom

    direction = jax.tree.map(_direction, g, mu, nu)
    mu = optax.tree.cast(mu, mu_dtype)
    return direction, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def ano(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.92,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    logarithmic_schedule: bool = False,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
  r"""ANO optimizer.

  ANO uses sign–magnitude decoupling (sign of momentum for direction, gradient
  magnitude for scaling) with an additive (Yogi-like) second-moment update.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    b1: First-moment decay β1.
    b2: Parameter for second-moment update β2.
    eps: Small constant ε added inside the square root.
    weight_decay: Decoupled weight decay coefficient.
    logarithmic_schedule: If True, use logarithmic
      schedule for b1: 1-1/log(max(2,k)).
    mu_dtype: Optional dtype for the first order accumulator m.

  Returns:
    The corresponding :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> solver = optax.contrib.ano(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kegreisz, `Ano: Faster is Better in Noisy Landscapes
    <https://zenodo.org/records/16933383>`_.
  """
  return combine.chain(
    scale_by_ano(
          b1=b1,
          b2=b2,
          eps=eps,
          logarithmic_schedule=logarithmic_schedule,
          mu_dtype=mu_dtype,
      ),
      transform.add_decayed_weights(weight_decay),
      transform.scale_by_learning_rate(learning_rate)
    )
