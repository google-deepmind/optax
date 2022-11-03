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

from typing import Any, Callable, Optional, Union

import jax.numpy as jnp

from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import privacy
from optax._src import transform
from optax._src import wrappers


ScalarOrSchedule = Union[float, base.Schedule]
MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)


def adabelief(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16) -> base.GradientTransformation:
  """The AdaBelief optimizer.

  AdaBelief is an adaptive learning rate optimizer that focuses on fast
  convergence, generalization, and stability. It adapts the step size depending
  on its "belief" in the gradient direction — the optimizer adaptively scales
  the step size by the difference between the predicted and observed gradients.
  AdaBelief is a modified version of Adam and contains the same number of
  parameters.

  References:
    Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

  Args:
    learning_rate: A fixed global scaling factor.
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
      _scale_by_learning_rate(learning_rate),
  )


def adafactor(
    learning_rate: Optional[ScalarOrSchedule] = None,
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

  References:
    Shazeer and Stern, 2018: https://arxiv.org/abs/1804.04235

  Args:
      learning_rate: A fixed global scaling factor. Note: the natural scale for
        Adafactor's LR is markedly different from Adam, one doesn't use the
        1/sqrt(hidden) correction for this optim with attention-based models.
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
    tx.append(_scale_by_learning_rate(learning_rate, flip_sign=False))
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
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7
) -> base.GradientTransformation:
  """The Adagrad optimizer.

  Adagrad is an algorithm for gradient based optimization that anneals the
  learning rate for each parameter during the course of training.

  WARNING: Adagrad's main limit is the monotonic accumulation of squared
  gradients in the denominator: since all terms are >0, the sum keeps growing
  during training and the learning rate eventually becomes vanishingly small.

  References:
    Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html

  Args:
    learning_rate: A fixed global scaling factor.
    initial_accumulator_value: Initial value for the accumulator.
    eps: A small constant applied to denominator inside of the square root
      (as in RMSProp) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def adam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""The classic Adam optimizer.

  Adam is an SGD variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectievly. The learning rate is
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
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: A fixed global scaling factor.
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
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      _scale_by_learning_rate(learning_rate),
  )


def adamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Adam with weight decay regularization.

  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: A fixed global scaling factor.
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
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )


def amsgrad(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """The AMSGrad optimiser.

  The original Adam can fail to converge to the optimal solution in some cases.
  AMSGrad guarantees convergence by using a long-term memory of past gradients.

  References:
    Reddi et al, 2018: https://openreview.net/forum?id=ryQu7f-RZ

  Args:
    learning_rate: A fixed global scaling factor.
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
      _scale_by_learning_rate(learning_rate),
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

  References:
    Bernstein et al, 2020: https://arxiv.org/abs/2002.03432

  Args:
    learning_rate: A fixed global scaling factor.
    min_norm: A minimum value that the norm of the gradient updates and the norm
      of the layer parameters can be clipped to to avoid dividing by zero when
      computing the trust ratio (as in the LARS paper).

  Returns:
    The corresponding `GradientTransformation`.
  """
  mult = 1 / jnp.sqrt(1 + learning_rate ** 2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      _scale_by_learning_rate(learning_rate * mult),
      transform.add_decayed_weights((mult - 1)),
  )


def lars(
    learning_rate: ScalarOrSchedule,
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

  References:
    You et al, 2017: https://arxiv.org/abs/1708.03888

  Args:
    learning_rate: A fixed global scaling factor.
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
      _scale_by_learning_rate(learning_rate),
      transform.trace(decay=momentum, nesterov=nesterov),
  )


def lamb(
    learning_rate: ScalarOrSchedule,
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

  References:
    You et al, 2019: https://arxiv.org/abs/1904.00962

  Args:
    learning_rate: A fixed global scaling factor.
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
      _scale_by_learning_rate(learning_rate),
  )


def noisy_sgd(
    learning_rate: ScalarOrSchedule,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0
) -> base.GradientTransformation:
  r"""A variant of SGD with added noise.

  It has been found that adding noise to the gradients can improve
  both the training error and the generalization error in very deep networks.

  References:
    Neelakantan et al, 2014: https://arxiv.org/abs/1511.06807

  Args:
    learning_rate: A fixed global scaling factor.
    eta: Initial variance for the Gaussian noise added to gradients.
    gamma: A parameter controlling the annealing of noise over time, the
      variance decays according to `(1+t)^-\gamma`.
    seed: Seed for the pseudo-random generation process.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.add_noise(eta, gamma, seed),
      _scale_by_learning_rate(learning_rate),
  )


def novograd(
    learning_rate: ScalarOrSchedule,
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

  References:
    Ginsburg et al, 2019: https://arxiv.org/abs/1905.11286
    Li et al, 2019: https://arxiv.org/abs/1904.03288

  Args:
    learning_rate: A fixed global scaling factor.
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
      _scale_by_learning_rate(learning_rate),
  )


def optimistic_gradient_descent(
    learning_rate: ScalarOrSchedule,
    alpha: ScalarOrSchedule = 1.0,
    beta: ScalarOrSchedule = 1.0
) -> base.GradientTransformation:
  """An Optimistic Gradient Descent optimizer.

  Optimistic gradient descent is an approximation of extra-gradient methods
  which require multiple gradient calls to compute the next update. It has
  strong formal guarantees for last-iterate convergence in min-max games, for
  which standard gradient descent can oscillate or even diverge.

  References:
    [Mokhtari et al, 2019](https://arxiv.org/abs/1901.08511v2)

  Args:
    learning_rate: A fixed global scaling factor.
    alpha: Coefficient for generalized OGD.
    beta: Coefficient for generalized OGD negative momentum.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_optimistic_gradient(alpha=alpha, beta=beta),
      _scale_by_learning_rate(learning_rate)
  )


def radam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0
) -> base.GradientTransformation:
  """The Rectified Adam optimizer.

  The adaptive learning rate in Adam has undesirably large variance in early
  stages of training, due to the limited number of training samples used to
  estimate the optimizer's statistics. Rectified Adam addresses this issue
  by analytically reducing the large variance.

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    threshold: Threshold for variance tractability.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_radam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, threshold=threshold),
      _scale_by_learning_rate(learning_rate),
  )


def rmsprop(
    learning_rate: ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.,
    centered: bool = False,
    momentum: Optional[float] = None,
    nesterov: bool = False
) -> base.GradientTransformation:
  # pylint: disable=line-too-long
  """A flexible RMSProp optimizer.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimizer that can be used to switch between several of these variants.

  References:
    Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
    Graves, 2013: https://arxiv.org/abs/1308.0850

  Args:
    learning_rate: A fixed global scaling factor.
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
        _scale_by_learning_rate(learning_rate),
        (transform.trace(decay=momentum, nesterov=nesterov)
         if momentum is not None else base.identity())
    )
  return combine.chain(
      transform.scale_by_rms(
          decay=decay, eps=eps, initial_scale=initial_scale),
      _scale_by_learning_rate(learning_rate),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else base.identity())
  )


def sgd(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """A canonical Stochastic Gradient Descent optimizer.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate: A fixed global scaling factor.
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    accumulator_dtype: Optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      (transform.trace(decay=momentum, nesterov=nesterov,
                       accumulator_dtype=accumulator_dtype)
       if momentum is not None else base.identity()),
      _scale_by_learning_rate(learning_rate)
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

  References:
    Anil et al, 2019: https://arxiv.org/abs/1901.11150

  Args:
    learning_rate: A fixed global scaling factor.
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
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
) -> base.GradientTransformation:
  """The Yogi optimizer.

  Yogi is an adaptive optimizer, which provides control in tuning the effective
  learning rate to prevent it from increasing. By doing so, it focuses on
  addressing the issues of convergence and generalization in exponential moving
  average-based adaptive methods (such as Adam and RMSprop). Yogi is a
  modification of Adam and uses the same parameters.

  References:
    Zaheer et al, 2020: http://www.sanjivk.com/yogi_nips2018.pdf

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def dpsgd(
    learning_rate: ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    momentum: Optional[float] = None,
    nesterov: bool = False
) -> base.GradientTransformation:
  """The DPSGD optimizer.

  Differential privacy is a standard for privacy guarantees of algorithms
  learning from aggregate databases including potentially sensitive information.
  DPSGD offers protection against a strong adversary with full knowledge of the
  training mechanism and access to the model’s parameters.

  WARNING: This `GradientTransformation` expects input updates to have a batch
  dimension on the 0th axis. That is, this function expects per-example
  gradients as input (which are easy to obtain in JAX using `jax.vmap`).

  References:
    Abadi et al, 2016: https://arxiv.org/abs/1607.00133

  Args:
    learning_rate: A fixed global scaling factor.
    l2_norm_clip: Maximum L2 norm of the per-example gradients.
    noise_multiplier: Ratio of standard deviation to the clipping norm.
    seed: Initial seed used for the jax.random.PRNGKey
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      privacy.differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else base.identity()),
      _scale_by_learning_rate(learning_rate)
  )


def adamax(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> base.GradientTransformation:
  """A variant of the Adam optimizer that uses the infinity norm.

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the maximum of past gradients.
    eps: A small constant applied to denominator to avoid dividing by zero when
      rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adamax(b1=b1, b2=b2, eps=eps,),
      _scale_by_learning_rate(learning_rate),
  )


def adamaxw(
    learning_rate: ScalarOrSchedule,
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

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: A fixed global scaling factor.
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
  """
  return combine.chain(
      transform.scale_by_adamax(b1=b1, b2=b2, eps=eps),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )
