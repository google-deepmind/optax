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
"""ADOPT (Adaptive Optimization with Provable Theoretical guarantees)."""

from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


def scale_by_adopt(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.9999,
    eps: jax.typing.ArrayLike = 1e-6,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable[[jax.Array], jax.Array] = lambda x: x**0.25,
) -> base.GradientTransformation:
  r"""Rescale updates according to the ADOPT algorithm.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.
    use_clipping: Whether to use gradient clipping to improve stability. When
      enabled, the clipping value is determined by the clip_value_fn.
    clip_value_fn: A function that takes a step index and returns a clipping
      value. Default is :math:`x^{0.25}`.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. seealso:: :func:`optax.contrib.adopt`
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    nu = optax.tree.zeros_like(params)  # Second moment
    return transform.ScaleByAdamState(
        count=jnp.zeros([], jnp.int32), mu=mu, nu=nu
    )

  def update_fn(updates, state, params=None):
    del params
    b2_ = jnp.where(state.count > 0, b2, 0)
    b1_ = jnp.where(state.count > 0, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2_, 2)
    if use_clipping:
      clip_value = clip_value_fn(state.count)
      mu_updates = jax.tree.map(
          lambda ud, nu: jnp.clip(
              ud / jnp.maximum(jnp.sqrt(nu), eps), -clip_value, clip_value
          ),
          updates,
          state.nu,
      )
    else:
      mu_updates = jax.tree.map(
          lambda ud, nu: ud / jnp.maximum(jnp.sqrt(nu), eps), updates, state.nu
      )
    mu = optax.tree.update_moment(mu_updates, state.mu, b1_, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_ = optax.tree.update_moment(mu_updates, mu, b1_, 1)
    else:
      mu_ = mu
    updates = mu_
    mu = optax.tree.cast(mu, mu_dtype)
    return updates, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def adopt(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.9999,
    eps: jax.typing.ArrayLike = 1e-6,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x**0.25,
) -> base.GradientTransformationExtraArgs:
  r"""ADOPT (Adaptive Optimization with Provable Theoretical guarantees).

  ADOPT is a modified version of Adam that may improve the robustness of Adam
  with respect to the choice of beta2. This implementation includes an optional
  clipping operation to improve stability, especially in early training stages.

  The key difference from Adam is that ADOPT modifies the update rule to avoid
  potential instability issues, particularly when some gradient elements are
  nearzero at initialization. With clipping enabled (default), ADOPT applies a
  clipping operation to improve stability, particularly in early training
  stages.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root to
      avoid dividing by zero when rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.
    use_clipping: Whether to apply clipping to improve stability. Recommended to
      keep as True, especially for training from scratch.
    clip_value_fn: A function that takes a step index and returns a clipping
      value. Default is :math:`x^{0.25}`.

  Returns:
    The corresponding :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.contrib.adopt(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
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
    Taniguchi et al, `ADOPT: Modified Adam Can Converge with Any beta2 with the
    Optimal Rate <https://arxiv.org/abs/2403.00855>`_, NeurIPS 2024
  """
  return combine.chain(
      scale_by_adopt(
          b1=b1,
          b2=b2,
          eps=eps,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
          use_clipping=use_clipping,
          clip_value_fn=clip_value_fn,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
