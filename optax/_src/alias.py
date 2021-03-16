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


@schedule.inject_hyperparams
def adabelief(learning_rate: ScalarOrSchedule,
              b1: ScalarOrSchedule = 0.9,
              b2: ScalarOrSchedule = 0.999,
              eps: ScalarOrSchedule = 1e-8) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_belief(b1=b1, b2=b2, eps=eps),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def adagrad(
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: ScalarOrSchedule = 1e-7) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def adam(learning_rate: ScalarOrSchedule,
         b1: ScalarOrSchedule = 0.9,
         b2: ScalarOrSchedule = 0.999,
         eps: ScalarOrSchedule = 1e-8,
         eps_root: ScalarOrSchedule = 0.0) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def adamw(learning_rate: ScalarOrSchedule,
          b1: ScalarOrSchedule = 0.9,
          b2: ScalarOrSchedule = 0.999,
          eps: ScalarOrSchedule = 1e-8,
          eps_root: ScalarOrSchedule = 0.0,
          weight_decay: ScalarOrSchedule = 1e-4) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.additive_weight_decay(weight_decay),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def fromage(learning_rate: ScalarOrSchedule,
            min_norm: ScalarOrSchedule = 1e-6) -> GradientTransformation:
  mult = 1 / jnp.sqrt(1 + learning_rate ** 2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      transform.scale(-learning_rate),
      transform.add_decayed_weights((mult - 1)),
  )


@schedule.inject_hyperparams
def lamb(learning_rate: ScalarOrSchedule,
         b1: ScalarOrSchedule = 0.9,
         b2: ScalarOrSchedule = 0.999,
         eps: ScalarOrSchedule = 1e-6,
         eps_root: ScalarOrSchedule = 0.0,
         weight_decay: ScalarOrSchedule = 0.) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay),
      transform.scale_by_trust_ratio(),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def noisy_sgd(learning_rate: ScalarOrSchedule,
              eta: ScalarOrSchedule = 0.01,
              gamma: ScalarOrSchedule = 0.55,
              seed: int = 0) -> GradientTransformation:
  return combine.chain(
      transform.trace(decay=0., nesterov=False),
      transform.scale(-learning_rate),
      transform.add_noise(eta, gamma, seed),
  )


@schedule.inject_hyperparams
def radam(learning_rate: ScalarOrSchedule,
          b1: ScalarOrSchedule = 0.9,
          b2: ScalarOrSchedule = 0.999,
          eps: ScalarOrSchedule = 1e-8,
          threshold: ScalarOrSchedule = 5.0) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_radam(b1=b1, b2=b2, eps=eps, threshold=threshold),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def rmsprop(learning_rate: ScalarOrSchedule,
            decay: ScalarOrSchedule = 0.9,
            eps: ScalarOrSchedule = 1e-8,
            centered: bool = False) -> GradientTransformation:
  if centered:
    return combine.chain(
        transform.scale_by_stddev(decay=decay, eps=eps),
        transform.scale(-learning_rate),
    )
  return combine.chain(
      transform.scale_by_rms(decay=decay, eps=eps),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def sgd(learning_rate: ScalarOrSchedule,
        momentum: ScalarOrSchedule = 0.,
        nesterov: bool = False) -> GradientTransformation:
  return combine.chain(
      transform.trace(decay=momentum, nesterov=nesterov),
      transform.scale(-learning_rate),
  )


@schedule.inject_hyperparams
def yogi(learning_rate: ScalarOrSchedule,
         b1: ScalarOrSchedule = 0.9,
         b2: ScalarOrSchedule = 0.999,
         eps: ScalarOrSchedule = 1e-3) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      transform.scale(-learning_rate),
  )
