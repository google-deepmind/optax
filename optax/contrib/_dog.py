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
"""Distance over gradients algorithm and its variants.

References:
  Ivgi et al., `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
  Size Schedule<https://arxiv.org/abs/2302.12022>`_, 2023.

  Khaled et al., `DoWG Unleashed: An Efficient Universal Parameter-Free
  Gradient Descent Method<https://arxiv.org/pdf/2305.16284>`_, 2023.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union, Literal

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform
import optax.tree


class DoGState(NamedTuple):
  """State for DoG optimizer."""

  is_init_step: jax.Array  # bool
  init_params: chex.ArrayTree
  max_dist: jax.Array
  sum_sq_norm_grads: jax.Array


def scale_by_dog(
    init_step: tuple[Literal["distance", "learning_rate", "heuristic"],
                     jax.typing.ArrayLike],
    eps: jax.typing.ArrayLike = 1e-8,
) -> base.GradientTransformation:
  r"""Scale by Distance over Gradients (DoG).

  See :func:`optax.contrib.dog` for more details.

  Args:
    init_step: Initial step specification.
    eps: Epsilon used for numerical stability.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3

  .. warning::
    The authors recommend using model averaging with this optimizer.

    This optimizer's ``init`` function should receive the actual parameters (not
    just dummy parameters) when the ``heuristic`` initial step is used.
  """

  init_step_type, init_step_value = init_step

  def init_fn(params: base.Params) -> DoGState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = optax.tree.dtype(params, "lowest")

    if init_step_type == "distance":
      r_epsilon = init_step_value
    elif init_step_type == "heuristic":
      r_epsilon = init_step_value * (1 + optax.tree.norm(params))
    elif init_step_type == "learning_rate":
      r_epsilon = 0.0
    else:
      raise ValueError(
          f"Invalid init_step specification for scale_by_dog: {init_step_type=}"
      )

    return DoGState(
        is_init_step=jnp.asarray(True),
        init_params=params,
        max_dist=jnp.asarray(r_epsilon, dtype=params_dtype),
        sum_sq_norm_grads=jnp.asarray(0.0, dtype=params_dtype),
    )

  def update_fn(
      updates: base.Updates, state: DoGState, params: base.Params
  ) -> tuple[base.Updates, DoGState]:
    dist = optax.tree.norm(optax.tree.sub(state.init_params, params))
    max_dist = jnp.maximum(state.max_dist, dist)
    sum_sq_norm_grads = state.sum_sq_norm_grads + optax.tree.norm(
        updates, squared=True
    )
    learning_rate = max_dist / jnp.sqrt(sum_sq_norm_grads + eps)

    if init_step_type == "learning_rate":
      learning_rate = jnp.where(
          state.is_init_step, init_step_value, learning_rate
      )

    new_updates = optax.tree.scale(learning_rate, updates)
    return new_updates, DoGState(
        is_init_step=jnp.asarray(False),
        init_params=state.init_params,
        max_dist=max_dist,
        sum_sq_norm_grads=sum_sq_norm_grads,
    )

  return base.GradientTransformation(init_fn, update_fn)


def dog(
    learning_rate: base.ScalarOrSchedule = 1.0,
    init_step: tuple[
        Literal["distance", "learning_rate", "heuristic"], jax.typing.ArrayLike
    ] = ("heuristic", 1e-6),
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: Optional[jax.typing.ArrayLike] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
):
  r"""Distance over Gradients (DoG) optimizer.

  DoG updates parameters :math:`x_t` with stochastic gradients :math:`g_t`
  according to the update rule:

  .. math::

    \begin{align*}
      r_t &= \| x_t - x_0 \| \\
      \bar{r}_t &= \max_{k \leq t} r_k \\
      G_t &= \sum_{k \leq t} \|g_k\|^2 \\
      \eta_t &= \frac{\bar{r}_t}{\sqrt{G_t + \epsilon}} \\
      x_{t+1} & = x_{t} - \eta_t\, g_t
    \end{align*}

  Args:
    learning_rate: optional learning rate (potentially varying according to
      some predetermined scheduler).
    init_step: Initial step specification. Consists of a pair ``(tag, value)``,
      where ``value`` is a float and ``tag`` is a string, which must be one of
      ``distance``, ``learning_rate``, or ``heuristic``.
      ``distance`` sets the initial distance :math:`r_0` (:math:`r_\epsilon` in
      the paper) to the given value.
      ``learning_rate`` sets the initial learning rate :math:`\eta_0` to the
      given value.
      ``heuristic`` sets  :math:`r_0 = \alpha (1 + \|x_0\|)`, where
      :math:`\alpha` is the given value. The suggested value of :math:`\alpha`
      is 1e-6, unless the model uses batch normalization, in which case the
      suggested value is 1e-4.
      As discussed in the paper, the value should be small enough to ensure that
      the initial update step will be small enough to not cause the model to
      diverge.
    eps: epsilon used for numerical stability - added to the sum of squared
      norm of gradients.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the gradient transformations is applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.dog()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  updates, opt_state = solver.update(
    ...    grad, opt_state, params, value=value)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: ', f(params))
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...

  References:
    Ivgi et al., `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
    Size Schedule <https://arxiv.org/abs/2302.12022>`_, 2023.

  .. versionadded:: 0.2.3

  .. warning::
    The authors recommend using model averaging with this optimizer.

    This optimizer's ``init`` function should receive the actual parameters (not
    just dummy parameters) when the ``heuristic`` initial step is used.
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask)
      if weight_decay is not None
      else base.identity(),
      scale_by_dog(init_step, eps),
      transform.scale_by_learning_rate(learning_rate),
  )


class DoWGState(NamedTuple):
  """State for DoWG optimizer."""

  init_params: chex.ArrayTree
  weighted_sq_norm_grads: jax.Array
  estim_sq_dist: jax.Array


def scale_by_dowg(
    init_estim_sq_dist: Optional[jax.typing.ArrayLike] = None,
    eps: jax.typing.ArrayLike = 1e-4,
) -> base.GradientTransformation:
  """Scale by Distance over Weighted Gradients (DoWG).

  See :func:`optax.contrib.dowg` for more details.

  Args:
    init_estim_sq_dist: initial guess of the squared distance to solution.
    eps: small value to prevent division by zero in the denominator definining,
      the learning rate, also used as initial guess for the distance to solution
      if ``init_estim_sq_dist`` is None.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3
  """

  def init_fn(params: base.Params) -> DoWGState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = optax.tree.dtype(params, "lowest")
    if init_estim_sq_dist is None:
      init_estim_sq_dist_ = eps
    else:
      init_estim_sq_dist_ = init_estim_sq_dist
    return DoWGState(
        init_params=params,
        estim_sq_dist=jnp.asarray(init_estim_sq_dist_, dtype=params_dtype),
        weighted_sq_norm_grads=jnp.asarray(0.0, dtype=params_dtype),
    )

  def update_fn(
      updates: base.Updates, state: DoWGState, params: base.Params
  ) -> tuple[base.Updates, DoWGState]:
    curr_sq_dist = optax.tree.norm(
        optax.tree.sub(state.init_params, params), squared=True
    )
    estim_sq_dist = jnp.maximum(state.estim_sq_dist, curr_sq_dist)
    step_sq_norm_grads = optax.tree.norm(updates, squared=True)
    weighted_sq_norm_grads = (
        estim_sq_dist * step_sq_norm_grads + state.weighted_sq_norm_grads
    )
    learning_rate = estim_sq_dist / (jnp.sqrt(weighted_sq_norm_grads) + eps)

    new_updates = optax.tree.scale(learning_rate, updates)
    return new_updates, state._replace(
        estim_sq_dist=estim_sq_dist,
        weighted_sq_norm_grads=weighted_sq_norm_grads,
    )

  return base.GradientTransformation(init_fn, update_fn)


def dowg(
    learning_rate: base.ScalarOrSchedule = 1.0,
    init_estim_sq_dist: Optional[jax.typing.ArrayLike] = None,
    eps: jax.typing.ArrayLike = 1e-4,
    weight_decay: Optional[jax.typing.ArrayLike] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
):
  r"""Distance over weighted Gradients optimizer.

  Examples:
    >>> import optax
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.dowg()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  updates, opt_state = solver.update(
    ...    grad, opt_state, params, value=value)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: ', f(params))
    Objective function:  13.925367
    Objective function:  13.872763
    Objective function:  13.775433
    Objective function:  13.596172
    Objective function:  13.268837

  References:
    Khaled et al., `DoWG Unleashed: An Efficient Universal Parameter-Free
    Gradient Descent Method <https://arxiv.org/pdf/2305.16284>`_, 2023.

  Args:
    learning_rate: optional learning rate (potentially varying according to some
      predetermined scheduler).
    init_estim_sq_dist: initial guess of the squared distance to solution.
    eps: small value to prevent division by zero in the denominator definining,
      the learning rate, also used as initial guess for the distance to solution
      if ``init_estim_sq_dist`` is None.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      weight decay to, and `False` for those you want to skip. Note that the
      gradient transformations is applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask)
      if weight_decay is not None
      else base.identity(),
      scale_by_dowg(init_estim_sq_dist, eps),
      transform.scale_by_learning_rate(learning_rate),
  )
