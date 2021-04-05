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

from typing import Union, Optional

import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import privacy
from optax._src import transform


ScalarOrSchedule = Union[float, base.Schedule]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule):
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: -learning_rate(count))
  return transform.scale(-learning_rate)


def adabelief(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8) -> base.GradientTransformation:
  """The AdaBelief optimiser.

  AdaBelief is an adaptive learning rate optimiser that focuses on fast
  convergence, generalisation, and stability. It adapts the step size depending
  on its "belief" in the gradient direction — the optimiser adaptively scales
  the step size by the difference between the predicted and observed gradients.
  AdaBelief is a modified version of Adam and contains the same number of
  parameters.

  References:
    [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

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
      transform.scale_by_belief(b1=b1, b2=b2, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


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
    [Duchi et al, 2011](https://jmlr.org/papers/v12/duchi11a.html)

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
    eps_root: float = 0.0
) -> base.GradientTransformation:
  """The classic Adam optimiser.

  Adam is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is computed from estimates of first- and second-order
  moments of the gradients (using suitable exponential moving averages).

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for example when computing (meta-)gradients through Adam.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      _scale_by_learning_rate(learning_rate),
  )


def adamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-4
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
    [Loshchilov et al, 2019](https://arxiv.org/abs/1711.05101)

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    weight_decay: strength of the weight decay regularization.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.additive_weight_decay(weight_decay),
      _scale_by_learning_rate(learning_rate),
  )


def fromage(
    learning_rate: float,
    min_norm: float = 1e-6
) -> base.GradientTransformation:
  mult = 1 / jnp.sqrt(1 + learning_rate ** 2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      _scale_by_learning_rate(learning_rate * mult),
      transform.add_decayed_weights((mult - 1)),
  )


def lamb(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.
) -> base.GradientTransformation:
  """The LAMB optimiser.

  LAMB is a general purpose layer-wise adaptive large batch optimiser designed
  to provide consistent training performance across a wide range of tasks,
  including those that use attention-based models (such as Transformers) and
  ResNet-50. The optimiser is able to work with small and large batches sizes.
  LAMB was inspired by the LARS learning algorithm.

  References:
    [You et al, 2019](https://arxiv.org/abs/1904.00962)
  
  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0.0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    weight_decay (default `0.`): strength of the weight decay regularization.    

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay),
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
    [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

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
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

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
  """A flexible RMSProp optimiser.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimiser that can be used to switch between several of these variants.

  References:
    [Tieleman and Hinton, 2012](
        www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    [Graves, 2013](https://arxiv.org/abs/1308.0850)

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
    nesterov: bool = False
) -> base.GradientTransformation:
  """A canonical Stochastic Gradient Descent optimiser.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    [Sutskever et al, 2013](http://proceedings.mlr.press/v28/sutskever13.pdf)

  Args:
    learning_rate: this is a fixed global scaling factor.
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else base.identity()),
      _scale_by_learning_rate(learning_rate)
  )


def yogi(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3
) -> base.GradientTransformation:
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
    [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

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
