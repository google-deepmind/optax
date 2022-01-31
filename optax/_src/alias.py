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
"""Aliases for popular optimisers."""

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
  """The AdaBelief optimiser.

  AdaBelief is an adaptive learning rate optimiser that focuses on fast
  convergence, generalisation, and stability. It adapts the step size depending
  on its "belief" in the gradient direction — the optimiser adaptively scales
  the step size by the difference between the predicted and observed gradients.
  AdaBelief is a modified version of Adam and contains the same number of
  parameters.

  References:
    Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    the corresponding `GradientTransformation`.
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
  """The Adafactor optimiser.

  Adafactor is an adaptive learning rate optimiser that focuses on fast
  training of large scale neural networks. It saves memory by using a factored
  estimate of the second order moments used to scale gradients.

  References:
    Shazeer and Stern, 2018: https://arxiv.org/abs/1804.04235

  Args:
      learning_rate: (float) a step size. Note: the natural scale for
        Adafactor's LR is markedly different from Adam, one doesn't use the
        1/sqrt(hidden) correction for this optim with attention-based models.
      min_dim_size_to_factor: (int) only factor the statistics if two array
        dimensions have at least this size.
      decay_rate: (float) controls second-moment exponential decay schedule.
      decay_offset: (int) for finetuning, one may set this to the starting
        step number of the finetuning phase.
      multiply_by_parameter_scale: (bool): if True, then scale learning_rate by
        parameter norm. if False, provided learning_rate is absolute step size.
      clipping_threshold: (float>=1) optional value; if None, clipping disabled.
      momentum: (float) optional value between 0 and 1, enables
        momentum and uses extra memory if non-None! None by default.
      dtype_momentum: (dtype) dtype of momentum buffers.
      weight_decay_rate: (float) optional rate at which to decay weights.
      eps: (float) regularization constant for root mean squared gradient.
      factored: (bool) whether to use factored second-moment estimates.
      weight_decay_mask: a tree with same structure as (or a prefix of)
        the params PyTree, or a Callable that returns such a pytree given
        the params/updates. The leaves should be booleans, `True`
        for leaves/subtrees you want to apply the transformation to,
        and `False` for those you want to skip.

  Returns:
    the corresponding `GradientTransformation`.
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

  Adagrad is an algorithm for gradient based optimisation that anneals the
  learning rate for each parameter during the course of training.

  WARNING: Adagrad's main limit is the monotonic accumulation of squared
  gradients in the denominator: since all terms are >0, the sum keeps growing
  during training and the learning rate eventually becomes vanishingly small.

  References:
    Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html

  Args:
    learning_rate: this is a fixed global scaling factor.
    initial_accumulator_value: initialisation for the accumulator.
    eps: a small constant applied to denominator inside of the square root
      (as in RMSProp) to avoid dividing by zero when rescaling.

  Returns:
    the corresponding `GradientTransformation`.
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
  """The classic Adam optimiser.

  Adam is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is computed from estimates of first- and second-order
  moments of the gradients (using suitable exponential moving averages).

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for example when computing (meta-)gradients through Adam.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    the corresponding `GradientTransformation`.
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

  AdamW uses weight decay to regularise learning towards small weights, as
  this leads to better generalisation. In SGD you can also use L2 regularisation
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
  for the bias parameters. You can use `optax.masked` to make your own AdamW
  variant where `additive_weight_decay` is applied only to a subset of `params`.

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: strength of the weight decay regularization.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )


def fromage(
    learning_rate: float,
    min_norm: float = 1e-6
) -> base.GradientTransformation:
  """The Frobenius matched gradient descent (Fromage) optimiser.

  Fromage is a learning algorithm that does not require learning rate tuning.
  The optimiser is based on modelling neural network gradients via deep relative
  trust (a distance function on deep neural networks). Fromage is similar to the
  LARS optimiser and can work on a range of standard neural network benchmarks,
  such as natural language Transformers and generative adversarial networks.

  References:
    Bernstein et al, 2020: https://arxiv.org/abs/2002.03432

  Args:
    learning_rate: this is a fixed global scaling factor.
    min_norm: a minimum value that the norm of the gradient updates and the
    norm of the layer parameters can be clipped to to avoid dividing by zero
    when computing the trust ratio (as in the LARS paper).

  Returns:
    the corresponding `GradientTransformation`.
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
  """The LARS optimiser.

  LAMB is a layer-wise adaptive optimiser introduced to help scale SGD to
  larger batch sizes. LARS later inspired the LAMB optimiser.

  References:
    You et al, 2017: https://arxiv.org/abs/1708.03888

  Args:
    learning_rate: this is a fixed global scaling factor.
    weight_decay (default `0.`): strength of the weight decay regularization.
    weight_decay_mask: a tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    trust_coefficient: a multiplier for the trust ratio.
    eps: optional additive constant in the trust ratio denominator.
    trust_ratio_mask: a tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    momentum: the decay rate for momentum.
    nesterov: whether to use Nesterov momentum.

  Returns:
    the corresponding `GradientTransformation`.
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
  """The LAMB optimiser.

  LAMB is a general purpose layer-wise adaptive large batch optimiser designed
  to provide consistent training performance across a wide range of tasks,
  including those that use attention-based models (such as Transformers) and
  ResNet-50. The optimiser is able to work with small and large batch sizes.
  LAMB was inspired by the LARS learning algorithm.

  References:
    You et al, 2019: https://arxiv.org/abs/1904.00962

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0.0`), a small constant applied to denominator inside
      the square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    weight_decay (default `0.`): strength of the weight decay regularization.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    the corresponding `GradientTransformation`.
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
  both the training error and the generalisation error in very deep networks.

  References:
    Neelakantan et al, 2014: https://arxiv.org/abs/1511.06807

  Args:
    learning_rate: this is a fixed global scaling factor.
    eta: the initial variance for the gaussian noise added to gradients.
    gamma: a parameter controlling the annealing of noise over time,
      the variance decays according to `(1+t)^-\gamma`.
    seed: the seed for the pseudo-random generation process.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      _scale_by_learning_rate(learning_rate),
      transform.add_noise(eta, gamma, seed),
  )


def radam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0
) -> base.GradientTransformation:
  """The Rectified Adam optimiser.

  The adaptive learning rate in Adam has undesirably large variance in early
  stages of training, due to the limited number of training samples used to
  estimate the optimiser's statistics. Rectified Adam addresses this issue
  by analytically reducing the large variance.

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    threshold: the threshold for variance tractability.

  Returns:
    the corresponding `GradientTransformation`.
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
  """A flexible RMSProp optimiser.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimiser that can be used to switch between several of these variants.

  References:
    Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
    Graves, 2013: https://arxiv.org/abs/1308.0850

  Args:
    learning_rate: this is a fixed global scaling factor.
    decay: the decay used to track the magnitude of previous gradients.
    eps: a small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: (default `0.`), initialisation of accumulators tracking the
      magnitude of previous updates. PyTorch uses `0`, TF1 uses `1`. When
      reproducing results from a paper, verify the value used by the authors.
    centered: (default `False`), whether the second moment or the variance of
      the past gradients is used to rescale the latest gradients.
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.

  Returns:
    the corresponding `GradientTransformation`.
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
  """A canonical Stochastic Gradient Descent optimiser.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate: this is a fixed global scaling factor.
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
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
  """The SM3 optimiser.

  SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients Method) is a
  memory-efficient adaptive optimiser designed to decrease memory overhead when
  training very large models, such as the Transformer for machine translation,
  BERT for language modelling, and AmoebaNet-D for image classification. SM3: 1)
  applies to tensors of arbitrary dimensions and any predefined cover of the
  parameters; 2) adapts the learning rates in an adaptive and data-driven manner
  (like Adagrad and unlike Adafactor); and 3) comes with rigorous convergence
  guarantees in stochastic convex optimization settings.

  References:
    Anil et al, 2019: https://arxiv.org/abs/1901.11150

  Args:
    learning_rate: this is a fixed global scaling factor.
    momentum: the `decay` rate used by the momentum term (when it is not set to
      `None`, then momentum is not used at all).

  Returns:
    the corresponding `GradientTransformation`.
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
  """The Yogi optimiser.

  Yogi is an adaptive optimiser, which provides control in tuning the effective
  learning rate to prevent it from increasing. By doing so, it focuses on
  addressing the issues of convergence and generalisation in exponential moving
  average-based adaptive methods (such as Adam and RMSprop). Yogi is a
  modification of Adam and uses the same parameters.

  References:
    Zaheer et al, 2020: http://www.sanjivk.com/yogi_nips2018.pdf

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    the corresponding `GradientTransformation`.
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
  """The DPSGD optimiser.

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
    learning_rate: this is a fixed global scaling factor.
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    seed: initial seed used for the jax.random.PRNGKey
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.

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
