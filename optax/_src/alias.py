# Lint as: python3
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

from optax._src import combine
from optax._src import privacy
from optax._src import schedule
from optax._src import transform


GradientTransformation = transform.GradientTransformation
ScalarOrSchedule = Union[float, schedule.Schedule]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule):
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: -learning_rate(count))
  return transform.scale(-learning_rate)


def adabelief(learning_rate: ScalarOrSchedule,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_belief(b1=b1, b2=b2, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def adagrad(
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def adam(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8,
         eps_root: float = 0.0) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      _scale_by_learning_rate(learning_rate),
  )


def adamw(learning_rate: ScalarOrSchedule,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          eps_root: float = 0.0,
          weight_decay: float = 1e-4) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.additive_weight_decay(weight_decay),
      _scale_by_learning_rate(learning_rate),
  )


def fromage(learning_rate: float,
            min_norm: float = 1e-6) -> GradientTransformation:
  mult = 1 / jnp.sqrt(1 + learning_rate ** 2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      _scale_by_learning_rate(learning_rate * mult),
      transform.add_decayed_weights((mult - 1)),
  )


def lamb(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-6,
         eps_root: float = 0.0,
         weight_decay: float = 0.) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay),
      transform.scale_by_trust_ratio(),
      _scale_by_learning_rate(learning_rate),
  )


def noisy_sgd(learning_rate: ScalarOrSchedule,
              eta: float = 0.01,
              gamma: float = 0.55,
              seed: int = 0) -> GradientTransformation:
  return combine.chain(
      _scale_by_learning_rate(learning_rate),
      transform.add_noise(eta, gamma, seed),
  )


def radam(learning_rate: ScalarOrSchedule,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          threshold: float = 5.0) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_radam(b1=b1, b2=b2, eps=eps, threshold=threshold),
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
) -> GradientTransformation:
  """A flexible RmsProp optimiser.

  RmsProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RmsProp can be found
  in the literature. This alias provides an easy to configure RmsProp
  optimiser that can be used to switch between several of these variants.

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
         if momentum is not None else transform.identity())
    )
  return combine.chain(
      transform.scale_by_rms(
          decay=decay, eps=eps, initial_scale=initial_scale),
      _scale_by_learning_rate(learning_rate),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else transform.identity())
  )


def sgd(learning_rate: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False) -> GradientTransformation:
  return combine.chain(
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else transform.identity()),
      _scale_by_learning_rate(learning_rate)
  )


def sm3(learning_rate: float,
        momentum: float = 0.9) -> GradientTransformation:
    return combine.chain(
        transform.scale_by_sm3(momentum),
        transform.scale(-learning_rate),
    )

def yogi(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-3) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def dpsgd(learning_rate: ScalarOrSchedule,
          l2_norm_clip: float,
          noise_multiplier: float,
          seed: int,
          momentum: Optional[float] = None,
          nesterov: bool = False) -> GradientTransformation:
  return combine.chain(
      privacy.differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else transform.identity()),
      _scale_by_learning_rate(learning_rate)
  )
