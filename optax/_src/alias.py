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

from typing import Union

import jax.numpy as jnp
from optax._src import combine
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
      transform.trace(decay=0., nesterov=False),
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


def rmsprop(learning_rate: ScalarOrSchedule,
            decay: float = 0.9,
            eps: float = 1e-8,
            centered: bool = False) -> GradientTransformation:
  if centered:
    return combine.chain(
        transform.scale_by_stddev(decay=decay, eps=eps),
        _scale_by_learning_rate(learning_rate),
    )
  return combine.chain(
      transform.scale_by_rms(decay=decay, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )


def sgd(learning_rate: ScalarOrSchedule,
        momentum: float = 0.,
        nesterov: bool = False) -> GradientTransformation:
  return combine.chain(
      transform.trace(decay=momentum, nesterov=nesterov),
      _scale_by_learning_rate(learning_rate),
  )


def yogi(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-3) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      _scale_by_learning_rate(learning_rate),
  )
