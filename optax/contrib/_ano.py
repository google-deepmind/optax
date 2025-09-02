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
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the ANO algorithm.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate parameter used in the sign-based second-moment update.
    eps: Term added to the denominator to improve numerical stability.
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

    # First-order moment: m_k = b1 * m_{k-1} + (1 - b1) * g_k
    mu = optax.tree.update_moment(g, state.mu, b1, 1)

    # Second moment with sign-based update:
    # v_k = v_{k-1} − (1 − b2) · sign(v_{k-1} − g_k^2) · g_k^2
    def _update_v(g_t, v_prev):
      g2 = jnp.square(g_t)
      return v_prev - (1.0 - b2) * jnp.sign(v_prev - g2) * g2

    nu = jax.tree.map(_update_v, g, state.nu)

    # Direction: |g| * sign(m) / (sqrt(v) + eps)
    def _direction(g_t, m_t, v_t):
      denom = jnp.sqrt(v_t) + eps
      return jnp.abs(g_t) * jnp.sign(m_t) / denom

    direction = jax.tree.map(_direction, g, mu, nu)
    count_inc = numerics.safe_increment(state.count)
    mu = optax.tree.cast(mu, mu_dtype)
    return direction, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def ano(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
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
    eps: Small constant ε added outside the square root.
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
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01

  References:
    Kegreisz, `Ano: Faster is Better in Noisy Landscapes
    <https://github.com/Adrienkgz/ano-optimizer>`_.
  """
  return combine.chain(
      scale_by_ano(
          b1=b1,
          b2=b2,
          eps=eps,
          mu_dtype=mu_dtype,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
