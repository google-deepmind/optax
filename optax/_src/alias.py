# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Aliases for popular optimizers."""

import functools
from typing import Any, Callable, Optional, Union

import jax.numpy as jnp

from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import linesearch as _linesearch
from optax._src import transform
from optax._src import wrappers


MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def adabelief(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16) -> base.GradientTransformation:
  r"""The AdaBelief optimizer.

  AdaBelief is an adaptive learning rate optimizer that focuses on fast
  convergence, generalization, and stability. It adapts the step size depending
  on its "belief" in the gradient direction — the optimizer adaptively scales
  the step size by the difference between the predicted and observed gradients.
  AdaBelief is a modified version of :func:`optax.adam` and contains the same
  number of parameters.

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, s_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      s_t &\leftarrow \beta_2 \cdot s_{t-1} + (1-\beta_2) \cdot (g_t - m_t)^2
      + \bar{\varepsilon} \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left(\sqrt{\hat{s}_t}
      + \varepsilon \right) \\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adabelief(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01

  References:
    Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_belief(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.scale_by_learning_rate(learning_rate),
  )


def adadelta(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    weight_decay_mask: MaskOrFn = None,
) -> base.GradientTransformation:
  """The Adadelta optimizer.

  Adadelta is a stochastic gradient descent method that adapts learning rates
  based on a moving window of gradient updates. Adadelta is a modification of
  Adagrad.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adadelta(learning_rate=10.)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.36E+01
    Objective function: 1.32E+01
    Objective function: 1.29E+01
    Objective function: 1.25E+01
    Objective function: 1.21E+01

  References:

    [Matthew D. Zeiler, 2012](https://arxiv.org/pdf/1212.5701.pdf)

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    rho: A coefficient used for computing a running average of squared
      gradients.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Optional rate at which to decay weights.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      transform.scale_by_adadelta(rho=rho, eps=eps),
      transform.scale_by_learning_rate(learning_rate),
  )


def adafactor(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: float = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: Any = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    weight_decay_mask: MaskOrFn = None,
    ) -> base.GradientTransformation:
  """The Adafactor optimizer.

  Adafactor is an adaptive learning rate optimizer that focuses on fast
  training of large scale neural networks. It saves memory by using a factored
  estimate of the second order moments used to scale gradients.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adafactor(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Shazeer and Stern, 2018: https://arxiv.org/abs/1804.04235

  Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        Note that the natural scale for Adafactor's LR is markedly different
        from Adam, one doesn't use the 1/sqrt(hidden) correction for this optim
        with attention-based models.
      min_dim_size_to_factor: Only factor the statistics if two array dimensions
        have at least this size.
      decay_rate: Controls second-moment exponential decay schedule.
      decay_offset: For fine-tuning, one may set this to the starting step
        number of the fine-tuning phase.
      multiply_by_parameter_scale: If True, then scale learning_rate by
        parameter norm. If False, provided learning_rate is absolute step size.
      clipping_threshold: Optional clipping threshold. Must be >= 1. If None,
        clipping is disabled.
      momentum: Optional value between 0 and 1, enables momentum and uses extra
        memory if non-None! None by default.
      dtype_momentum: Data type of momentum buffers.
      weight_decay_rate: Optional rate at which to decay weights.
      eps: Regularization constant for root mean squared gradient.
      factored: Whether to use factored second-moment estimates.
      weight_decay_mask: A tree with same structure as (or a prefix of)
        the params PyTree, or a Callable that returns such a pytree given
        the params/updates. The leaves should be booleans, `True`
        for leaves/subtrees you want to apply the transformation to,
        and `False` for those you want to skip.

  Returns:
    The corresponding `GradientTransformation`.
  """
  # The core of the algorithm is a procedure for rescaling gradients
  # by a factored estimate of the root mean squared gradients.
  # This reduces memory compared to algorithms such as Adam or RmsProp,
  # by not having to hold a separate estimate for each weight.
  tx = [
      factorized.scale_by_factored_rms(
          factored, decay_rate, decay_offset, min_dim_size_to_factor, eps)]
  # This basic rescaling is typically combined with one or more of the following
  # transformation (all can be disabled via adafactor's constructor args).
  if clipping_threshold is not None:
    tx.append(clipping.clip_by_block_rms(clipping_threshold))
  if learning_rate is not None:
    tx.append(transform.scale_by_learning_rate(learning_rate, flip_sign=False))
  if multiply_by_parameter_scale:
    tx.append(transform.scale_by_param_block_rms())
  if momentum is not None:
    tx.append(
        transform.ema(momentum, debias=False, accumulator_dtype=dtype_momentum))
  if weight_decay_rate is not None:
    tx.append(transform.add_decayed_weights(
        weight_decay_rate, mask=weight_decay_mask))
  # In gradient "descent" we follow the negative gradient.
  tx.append(transform.scale(-1))
  return combine.chain(*tx)


def adagrad(
    learning_rate: base.ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7
) -> base.GradientTransformation:
  """The Adagrad optimizer.

  Adagrad is an algorithm for gradient based optimization that anneals the
  learning rate for each parameter during the course of training.

  .. warning::
    Adagrad's main limit is the monotonic accumulation of squared
    gradients in the denominator: since all terms are >0, the sum keeps growing
    during training and the learning rate eventually becomes vanishingly small.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adagrad(learning_rate=1.0)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 5.01E+00
    Objective function: 2.40E+00
    Objective function: 1.25E+00
    Objective function: 6.86E-01
    Objective function: 3.85E-01

  References:
    Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    initial_accumulator_value: Initial value for the accumulator.
    eps: A small constant applied to denominator inside of the square root
      (as in RMSProp) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps),
      transform.scale_by_learning_rate(learning_rate),
  )


def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False
) -> base.GradientTransformation:
  r"""The Adam optimizer.

  Adam is an SGD variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adam(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kingma et al, `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_, 2014

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. warning::
    PyTorch and optax's implementation follow Algorithm 1 of [Kingma et al.
    2014]. Note that TensorFlow used instead the formulation just before Section
    2.1 of the paper. See https://github.com/deepmind/optax/issues/571 for more
    detail.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The solver with
      nesterov=True is equivalent to the :func:`optax.nadam` optimizer, and
      described in [Dozat 2016].

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.nadam`, :func:`optax.adamw`.
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


nadam = functools.partial(adam, nesterov=True)
nadam.__doc__ = (
    r"""The NAdam optimizer.

  Nadam is a variant of :func:`optax.adam` with Nesterov's momentum. The update
  rule of this solver is as follows:

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow
      \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}\\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.nadam(learning_rate=0.003)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  print('Objective function: {:.2E}'.format(f(params)))
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.38E+01
      Objective function: 1.38E+01

  References:
    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. versionadded:: 0.1.9

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.adam`, :func:`optax.nadamw`.
"""
)


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""Adam with weight decay regularization.

  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam,
  see [Loshchilov et al, 2019].

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function. Let :math:`\lambda` be the weight decay and
  :math:`\theta_t` the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t`
  and the parameters :math:`\theta_t` and computes updates :math:`u_t` and
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \left( \hat{m}_t / \left({\sqrt{\hat{v}_t
      + \bar{\varepsilon}} + \varepsilon} \right) + \lambda \theta_{t} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  This implementation can incorporate a momentum a la Nesterov introduced by
  [Dozat 2016]. The resulting optimizer is then often referred as NAdamW.
  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamw(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, `Decoupled Weight Decay
    Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.
    nesterov: Whether to use Nesterov momentum. The solver with
      nesterov=True is equivalent to the :func:`optax.nadamw` optimizer. This
      modification is described in [Dozat 2016].

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.adam`, :func:`optax.nadamw`.
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


nadamw = functools.partial(adamw, nesterov=True)
nadamw.__doc__ = (
    r"""NAdamW optimizer, implemented as part of the AdamW optimizer.

  NadamW is variant of :func:`optax.adamw` with Nesterov's momentum. Compared
  to AdamW, this optimizer replaces the assignment

  .. math::

      \hat{m}_t \leftarrow m_t / {(1-\beta_1^t)}

  with

  .. math::

      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.nadamw(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, `Decoupled Weight Decay
    Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. versionadded:: 0.1.9

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.adam`, :func:`optax.adamw`.
"""
)


def lion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """The Lion optimizer.

  Lion is discovered by symbolic program search. Unlike most adaptive optimizers
  such as AdamW, Lion only tracks momentum, making it more memory-efficient.
  The update of Lion is produced through the sign operation, resulting in a
  larger norm compared to updates produced by other optimizers such as SGD and
  AdamW. A suitable learning rate for Lion is typically 3-10x smaller than that
  for AdamW, the weight decay for Lion should be in turn 3-10x larger than that
  for AdamW to maintain a similar strength (lr * wd).

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lion(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Chen et al, 2023: https://arxiv.org/abs/2302.06675

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Rate to combine the momentum and the current gradient.
    b2: Exponential decay rate to track the momentum of past gradients.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


def amsgrad(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """The AMSGrad optimiser.

  The original Adam can fail to converge to the optimal solution in some cases.
  AMSGrad guarantees convergence by using a long-term memory of past gradients.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.amsgrad(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Reddi et al, 2018: https://openreview.net/forum?id=ryQu7f-RZ

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_amsgrad(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.scale_by_learning_rate(learning_rate),
  )


def fromage(
    learning_rate: float,
    min_norm: float = 1e-6
) -> base.GradientTransformation:
  """The Frobenius matched gradient descent (Fromage) optimizer.

  Fromage is a learning algorithm that does not require learning rate tuning.
  The optimizer is based on modeling neural network gradients via deep relative
  trust (a distance function on deep neural networks). Fromage is similar to the
  LARS optimizer and can work on a range of standard neural network benchmarks,
  such as natural language Transformers and generative adversarial networks.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.fromage(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Bernstein et al, 2020: https://arxiv.org/abs/2002.03432

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    min_norm: A minimum value that the norm of the gradient updates and the norm
      of the layer parameters can be clipped to to avoid dividing by zero when
      computing the trust ratio (as in the LARS paper).

  Returns:
    The corresponding `GradientTransformation`.
  """
  mult = 1 / jnp.sqrt(1 + learning_rate ** 2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      transform.scale_by_learning_rate(learning_rate * mult),
      transform.add_decayed_weights((mult - 1)),
  )


def lars(
    learning_rate: base.ScalarOrSchedule,
    weight_decay: float = 0.,
    weight_decay_mask: MaskOrFn = True,
    trust_coefficient: float = 0.001,
    eps: float = 0.,
    trust_ratio_mask: MaskOrFn = True,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The LARS optimizer.

  LARS is a layer-wise adaptive optimizer introduced to help scale SGD to
  larger batch sizes. LARS later inspired the LAMB optimizer.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lars(learning_rate=0.003)
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
    You et al, 2017: https://arxiv.org/abs/1708.03888

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    weight_decay: Strength of the weight decay regularization.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Optional additive constant in the trust ratio denominator.
    trust_ratio_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    momentum: Decay rate for momentum.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      wrappers.masked(
          inner=transform.scale_by_trust_ratio(
              trust_coefficient=trust_coefficient, eps=eps),
          mask=trust_ratio_mask),
      transform.scale_by_learning_rate(learning_rate),
      transform.trace(decay=momentum, nesterov=nesterov),
  )


def lamb(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.,
    mask: MaskOrFn = None,
) -> base.GradientTransformation:
  """The LAMB optimizer.

  LAMB is a general purpose layer-wise adaptive large batch optimizer designed
  to provide consistent training performance across a wide range of tasks,
  including those that use attention-based models (such as Transformers) and
  ResNet-50. The optimizer is able to work with small and large batch sizes.
  LAMB was inspired by the LARS learning algorithm.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lamb(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    You et al, 2019: https://arxiv.org/abs/1904.00962

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay=weight_decay, mask=mask),
      transform.scale_by_trust_ratio(),
      transform.scale_by_learning_rate(learning_rate),
  )


def noisy_sgd(
    learning_rate: base.ScalarOrSchedule,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0
) -> base.GradientTransformation:
  r"""A variant of SGD with added noise.

  Noisy SGD is a variant of :func:`optax.sgd` that incorporates Gaussian noise
  into the updates. It has been found that adding noise to the gradients can
  improve both the training error and the generalization error in very deep
  networks.

  The update :math:`u_t` is modified to include this noise as follows:

  .. math::
    u_t \leftarrow -\alpha_t (g_t + N(0, \sigma_t^2)),

  where :math:`N(0, \sigma_t^2)` represents Gaussian noise with zero mean and a
  variance of :math:`\sigma_t^2`.

  The variance of this noise decays over time according to the formula

  .. math::
    \sigma_t^2 = \frac{\eta}{(1+t)^\gamma},

  where :math:`\gamma` is the decay rate parameter ``gamma`` and :math:`\eta`
  represents the initial variance ``eta``.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.noisy_sgd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Neelakantan et al, 2014: https://arxiv.org/abs/1511.06807

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    eta: Initial variance for the Gaussian noise added to gradients.
    gamma: A parameter controlling the annealing of noise over time ``t``, the
      variance decays according to ``(1+t)**(-gamma)``.
    seed: Seed for the pseudo-random generation process.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.add_noise(eta, gamma, seed),
      transform.scale_by_learning_rate(learning_rate),
  )


def novograd(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.,
) -> base.GradientTransformation:
  """NovoGrad optimizer.

  NovoGrad is more robust to the initial learning rate and
  weight initialization than other methods. For example,
  NovoGrad works well without LR warm-up, while other methods require it.
  NovoGrad performs exceptionally well for large batch training, e.g. it
  outperforms other methods for ResNet-50 for all batches up to 32K.
  In addition, NovoGrad requires half the memory compared to Adam.
  It was introduced together with Jasper ASR model.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.novograd(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01

  References:
    Ginsburg et al, 2019: https://arxiv.org/abs/1905.11286
    Li et al, 2019: https://arxiv.org/abs/1904.03288

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: An exponential decay rate to track the first moment of past gradients.
    b2: An exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside
      the square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    weight_decay: Strength of the weight decay regularization.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_novograd(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, weight_decay=weight_decay),
      transform.scale_by_learning_rate(learning_rate),
  )


def optimistic_gradient_descent(
    learning_rate: base.ScalarOrSchedule,
    alpha: base.ScalarOrSchedule = 1.0,
    beta: base.ScalarOrSchedule = 1.0
) -> base.GradientTransformation:
  """An Optimistic Gradient Descent optimizer.

  Optimistic gradient descent is an approximation of extra-gradient methods
  which require multiple gradient calls to compute the next update. It has
  strong formal guarantees for last-iterate convergence in min-max games, for
  which standard gradient descent can oscillate or even diverge.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.optimistic_gradient_descent(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01
    Objective function: 1.30E+01

  References:
    Mokhtari et al, 2019: https://arxiv.org/abs/1901.08511v2

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    alpha: Coefficient for generalized OGD.
    beta: Coefficient for generalized OGD negative momentum.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_optimistic_gradient(alpha=alpha, beta=beta),
      transform.scale_by_learning_rate(learning_rate)
  )


def radam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The Rectified Adam optimizer.

  The adaptive learning rate in Adam has undesirably large variance in early
  stages of training, due to the limited number of training samples used to
  estimate the optimizer's statistics. Rectified Adam addresses this issue
  by analytically reducing the large variance.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.radam(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Liu et al, 2020: https://arxiv.org/abs/1908.03265

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    threshold: Threshold for variance tractability.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_radam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          threshold=threshold,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def rmsprop(
    learning_rate: base.ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.,
    centered: bool = False,
    momentum: Optional[float] = None,
    nesterov: bool = False
) -> base.GradientTransformation:
  # pylint: disable=line-too-long
  r"""A flexible RMSProp optimizer.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimizer that can be used to switch between several of these variants.

  ..warning::
    PyTorch and optax's RMSprop implementations differ and could impact
    performance. In the denominator, optax uses :math:`$\sqrt{v + \epsilon}$`
    whereas PyTorch uses :math:`$\sqrt{v} + \epsilon$`. See
    https://github.com/google-deepmind/optax/issues/532 for more detail.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.rmsprop(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
    Graves, 2013: https://arxiv.org/abs/1308.0850

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    decay: Decay used to track the magnitude of previous gradients.
    eps: A small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: Initial value of accumulators tracking the magnitude of
      previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
      from a paper, verify the value used by the authors.
    centered: Whether the second moment or the variance of the past gradients is
      used to rescale the latest gradients.
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.

  Returns:
    The corresponding `GradientTransformation`.
  """
  # pylint: enable=line-too-long
  if centered:
    return combine.chain(
        transform.scale_by_stddev(
            decay=decay, eps=eps, initial_scale=initial_scale),
        transform.scale_by_learning_rate(learning_rate),
        (transform.trace(decay=momentum, nesterov=nesterov)
         if momentum is not None else base.identity())
    )
  return combine.chain(
      transform.scale_by_rms(
          decay=decay, eps=eps, initial_scale=initial_scale),
      transform.scale_by_learning_rate(learning_rate),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else base.identity())
  )


def sgd(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""A canonical Stochastic Gradient Descent optimizer.

  This implements stochastic gradient descent. It also includes support for
  momentum, and Nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.


  The canonical stochastic gradient descent returns an update
  :math:`u_t` of the form

  .. math::
    u_t \leftarrow -\alpha_t g_t,

  where :math:`g_t` is the gradient of the objective (potentially preprocessed
  by other transformations) and :math:`\alpha_t` is the ``learning_rate`` at
  time :math:`t` (constant or selected by an :class:`optax.Schedule`).

  Stochastic gradient descent with momentum takes two possible forms.

  .. math::

    \begin{align*}
      m_t &\leftarrow g_t + \mu m_{t-1} \\
      u_t &\leftarrow \begin{cases}
        -\alpha_t m_t & \text{ if } \texttt{nesterov = False} \\
        -\alpha_t (g_t + \mu m_t) & \text{ if } \texttt{nesterov = True}
        \end{cases} \\
      S_t &\leftarrow m_t,
    \end{align*}

  where :math:`\mu` is the ``momentum`` parameter and :math:`S_t` is the state
  of the optimizer.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.sgd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Sutskever et al, `On the importance of initialization and momentum in deep
    learning <http://proceedings.mlr.press/v28/sutskever13.pdf>`_, 2013

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    momentum: Decay rate used by the momentum term, when it is set to ``None``,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    accumulator_dtype: Optional ``dtype`` to be used for the accumulator; if
      ``None`` then the ``dtype`` is inferred from ``params`` and ``updates``.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      (transform.trace(decay=momentum, nesterov=nesterov,
                       accumulator_dtype=accumulator_dtype)
       if momentum is not None else base.identity()),
      transform.scale_by_learning_rate(learning_rate)
  )


def sm3(
    learning_rate: float,
    momentum: float = 0.9
) -> base.GradientTransformation:
  """The SM3 optimizer.

  SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients Method) is a
  memory-efficient adaptive optimizer designed to decrease memory overhead when
  training very large models, such as the Transformer for machine translation,
  BERT for language modeling, and AmoebaNet-D for image classification. SM3: 1)
  applies to tensors of arbitrary dimensions and any predefined cover of the
  parameters; 2) adapts the learning rates in an adaptive and data-driven manner
  (like Adagrad and unlike Adafactor); and 3) comes with rigorous convergence
  guarantees in stochastic convex optimization settings.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.sm3(learning_rate=0.003)
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
    Anil et al, 2019: https://arxiv.org/abs/1901.11150

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    momentum: Decay rate used by the momentum term (when it is not set to
      `None`, then momentum is not used at all).

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_sm3(momentum),
      transform.scale(-learning_rate),
  )


def yogi(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
) -> base.GradientTransformation:
  # pylint: disable=line-too-long
  """The Yogi optimizer.

  Yogi is an adaptive optimizer, which provides control in tuning the effective
  learning rate to prevent it from increasing. By doing so, it focuses on
  addressing the issues of convergence and generalization in exponential moving
  average-based adaptive methods (such as Adam and RMSprop). Yogi is a
  modification of Adam and uses the same parameters.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.yogi(learning_rate=0.002)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01

  References:
    Zaheer et al, 2018: https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  # pylint: enable=line-too-long
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      transform.scale_by_learning_rate(learning_rate),
  )


def adamax(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> base.GradientTransformation:
  r"""A variant of the Adam optimizer that uses the infinity norm.

  AdaMax is a variant of the :func:`optax.adam` optimizer. By generalizing
  Adam's :math:`L^2` norm to an :math:`L^p` norm and taking the limit as
  :math:`p \rightarrow \infty`, we obtain a simple and stable update rule.

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon` represent the arguments
  ``b1``, ``b2`` and ``eps`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \max(\left| g_t \right| + \varepsilon, \beta_2 \cdot
      v_{t-1}) \\
      \hat{m}_t &\leftarrow m_t / (1-\beta_1^t) \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / v_t \\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamax(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the maximum of past gradients.
    eps: A small constant applied to denominator to avoid dividing by zero when
      rescaling.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.adam`, :func:`optax.adamaxw`.
  """
  return combine.chain(
      transform.scale_by_adamax(b1=b1, b2=b2, eps=eps,),
      transform.scale_by_learning_rate(learning_rate),
  )


def adamaxw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Adamax with weight decay regularization.

  AdamaxW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
  for the bias parameters. You can use `optax.masked` to make your own AdamaxW
  variant where `additive_weight_decay` is applied only to a subset of `params`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamaxw(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the maximum of past gradients.
    eps: A small constant applied to denominator to avoid dividing by zero when
      rescaling.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adamax gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso:: :func:`optax.adam`, :func:`optax.adamax`.
  """
  return combine.chain(
      transform.scale_by_adamax(b1=b1, b2=b2, eps=eps),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


def rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> base.GradientTransformation:
  """The Rprop optimizer.

  Rprop, short for resillient backpropogation, is a first order variant of
  gradient descent. It responds only to the sign of the gradient by increasing
  or decreasing the step size selected per parameter exponentially to speed up
  convergence and avoid oscillations.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.rprop(learning_rate=0.003)
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
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Riedmiller and Braun. `A direct adaptive method for faster backpropagation
    learning: the RPROP algorithm
    <https://ieeexplore.ieee.org/document/298623>`_, 1993

    Igel and Hüsken.  `Empirical evaluation of the improved Rprop learning
    algorithms
    <https://www.sciencedirect.com/science/article/abs/pii/S0925231201007007>`_,
    2003

  Args:
    learning_rate: The initial step size.
    eta_minus: Multiplicative factor for decreasing step size. This is applied
      when the gradient changes sign from one step to the next.
    eta_plus: Multiplicative factor for increasing step size. This is applied
      when the gradient has the same sign from one step to the next.
    min_step_size: Minimum allowed step size. Smaller steps will be clipped to
      this value.
    max_step_size: Maximum allowed step size. Larger steps will be clipped to
      this value.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_rprop(
          learning_rate=learning_rate,
          eta_minus=eta_minus,
          eta_plus=eta_plus,
          min_step_size=min_step_size,
          max_step_size=max_step_size,
      ),
      transform.scale(-1.0),
  )


def polyak_sgd(
    max_learning_rate: float = 1.,
    scaling: base.ScalarOrSchedule = 1.,
    f_min: float = 0.0,
    eps: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  r"""SGD with Polyak step-size.

  This solver implements the SGD with Polyak step size of (Loizou et al. 2021).
  It sets the step-size as

  .. math::
    s \min\left\{\frac{f(x) - f^\star}{\|\nabla f(x)\|^2 + \epsilon},
      \gamma_{\max}\right\}\,,

  where :math:`f` is the function from which a gradient is computed,
  :math:`\gamma_{\max}` is a maximal acceptable learning rate set  by
  ``max_learning_rate``, :math:`\epsilon` is a constant preventing division by
  zero set with ``eps``, :math:`s` scales the formula by ``scaling``, and
  :math:`f^\star` is a guess of the minimum value of the function set with
  ``f_min``.


  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.polyak_sgd()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  params, opt_state = solver.update(grad, opt_state, params, value=value)
    ...  print('Objective function: ', f(params))
    Objective function:  3.5
    Objective function:  0.875
    Objective function:  0.21875
    Objective function:  0.0546875
    Objective function:  0.013671875

  .. warning::
    This method requires knowledge of an approximate value of the of the
    objective function minimum, passed through the ``f_min`` argument.
    For models that interpolate the data, this can be set to 0 (default
    value).
    Failing to set an appropriate value for ``f_min`` can lead to
    divergence or convergence to a suboptimal solution.

  References:
    Loizou et al. `Stochastic polyak step-size for SGD: An adaptive learning
    rate for fast convergence <https://arxiv.org/abs/2002.10542>`_, 2021

    Berrada et al., `Training neural networks for and by interpolation
    <https://arxiv.org/pdf/1906.05661.pdf>`_, 2020

  Args:
    max_learning_rate: a maximum step size to use (defaults to 1).
    scaling: A global scaling factor, either fixed or evolving along
      iterations with a scheduler (defaults to 1).
    f_min: a lower bound on the objective function (defaults to 0). Corresponds
      to :math:`f^\star` in the formula above.
    eps: a value to add in the denominator of the update (defaults to 0).

  Returns:
    A :class:`GradientTransformationExtraArgs`, where the ``update`` function
    takes an additional keyword argument ``value`` containing the current
    value of the objective function.
  """
  return combine.chain(
      sgd(learning_rate=scaling),
      transform.scale_by_polyak(
          max_learning_rate=max_learning_rate, f_min=f_min, eps=eps
      ),
  )


def lbfgs(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    linesearch: Optional[
        base.GradientTransformationExtraArgs
    ] = _linesearch.scale_by_backtracking_linesearch(
        max_backtracking_steps=30, store_grad=True
    ),
) -> base.GradientTransformationExtraArgs:
  r"""L-BFGS optimizer.

  L-BFGS is a quasi-Newton method that multiplies the update (gradient)
  with an approximation of the inverse Hessian. This algorithm doesn't need
  access to the Hessian, as this approximation is constructed from the gradient
  evaluations seen during optimization. L-BFGS is a limited-memory variant of
  the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm. The BFGS algorithm
  requires storing a matrix of size :math:`p \times p` with :math:`p` the
  dimension of the parameters.
  The limited variant circuments this issue by computing the approximation of
  the inverse using only :math:`m` (``memory_size``) past differences of
  parameters/gradients. Namely, the approximation of the Hessian inverse is
  denoted :math:`P_k = P_{k, k}`, where

  .. math::
    \begin{align*}
      P_{k, j+1} & = V_j^\top P_{k, j} V_j
        + \rho_j \delta w_j \delta w_j^\top
        \quad \mbox{for} \ j \in \{k-m, \ldots, k-1\}\\
      P_{k, k-m} & = \gamma_k I \\
      V_k & = I - \rho_k \delta u_k \delta w_k^\top \\
      \rho_k & = 1/(\delta u_k^\top \delta w_k) \\
      \delta w_k & = w_{k+1} - w_k \\
      \delta u_k & = u_{k+1} - u_k \\
      \gamma_k & =
        \begin{cases}
          (\delta w_{k-1}^\top \delta u_{k-1}) /
          (\delta u_{k-1}^\top \delta u_{k-1})
          & \mbox{if} \ \texttt{scale\_init\_hess} \\
          1 & \mbox{otherwise}
        \end{cases},
    \end{align*}

  for :math:`u_k` the gradients/updates at iteration :math:`k`, :math:`w_k` the
  parameters at iteration :math:`k`.

  The formula for updating :math:`P_k` is obtained by computing the optimal
  preconditioning matrix subject to some secant condition, see references
  for more details. Computing :math:`P_k u_k` can be done by a sequence of
  vector operations using past differences of parameters and gradients stored in
  a memory bufffer.

  The present function just outputs the LBFGS direction :math:`P_k u_k`.
  It can be chained with a linesearch ensuring sufficient decrease and low
  curvature, such as a zoom linesearch. The linesearch computes a stepsize
  :math:`\eta_k`, such that the updated parameters
  (using :func:`optax.apply_updates`) take the form
  :math:`w_{k+1} = w_k - \eta_k P_k u_k`.

  Example:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> solver = optax.lbfgs()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> value_and_grad = optax.value_and_grad_from_state(f)
    >>> for _ in range(5):
    ...   value, grad = value_and_grad(params, state=opt_state)
    ...   updates, opt_state = solver.update(
    ...      grad, opt_state, params, value=value, grad=grad, value_fn=f
    ...   )
    ...   params = optax.apply_updates(params, updates)
    ...   print('Objective function: ', f(params))
    Objective function:  5.040001
    Objective function:  7.460699e-14
    Objective function:  1.0602291e-27
    Objective function:  0.0
    Objective function:  0.0

  .. warning::
    This method is memory intensive optimizer, best used for small to medium
    scale problems.

  .. warning::
    This optimizer works best with a linesearch (current default is a
    backtracking linesearch). See example above for best use in a non-stochastic
    setting, where we can recycle gradients computed by the linesearch using
    :func:`optax.value_and_grad_from_state`.

  References:
    Algorithms 7.4, 7.5 (page 199) of Nocedal et al, `Numerical Optimization
    <https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf>_`
    , 1999

    Liu et al., `On the limited memory BFGS method for large scale optimization
    <https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf>`_
    , 1989.

  Args:
    learning_rate: optional global scaling factor, either fixed or evolving
      along iterations with a scheduler, see
      :func:`optax.scale_by_learning_rate`. By default the learning rate is
      handled by a linesearch.
    memory_size: number of past updates to keep in memory to approximate the
      Hessian inverse.
    scale_init_precond: whether to use a scaled identity as the initial
      preconditioner, see formula above.
    linesearch: an instance of :class:`optax.GradientTransformationExtraArgs`
      such as :func:`optax.scale_by_backtracking_linesearch` that computes a
      learning rate, a.k.a. stepsize, to satisfy some criterion such as a
      sufficient decrease of the objective by additional calls to the objective.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs` object.
  """
  if learning_rate is None:
    base_scaling = transform.scale(-1.0)
  else:
    base_scaling = transform.scale_by_learning_rate(learning_rate)
  if linesearch is None:
    linesearch = base.identity()
  return combine.chain(
      transform.scale_by_lbfgs(
          memory_size=memory_size, scale_init_precond=scale_init_precond
      ),
      base_scaling,
      linesearch,
  )
