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

from optax._src import combine
from optax._src import transform
GradientTransformation = transform.GradientTransformation


def adabelief(learning_rate: float,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_belief(b1=b1, b2=b2, eps=eps),
      transform.scale(-learning_rate),
  )


def adagrad(
    learning_rate: float,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps),
      transform.scale(-learning_rate),
  )


def adam(learning_rate: float,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps),
      transform.scale(-learning_rate),
  )


def adamw(learning_rate: float,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          weight_decay: float = 1e-4) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps),
      transform.additive_weight_decay(weight_decay),
      transform.scale(-learning_rate),
  )


def lamb(learning_rate: float,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-6,
         weight_decay: float = 0.) -> GradientTransformation:
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps),
      transform.additive_weight_decay(weight_decay),
      transform.scale_by_trust_ratio(),
      transform.scale(-learning_rate),
  )


def noisy_sgd(learning_rate: float,
              eta: float = 0.01,
              gamma: float = 0.55,
              seed: int = 0) -> GradientTransformation:
  return combine.chain(
      transform.trace(decay=0., nesterov=False),
      transform.scale(-learning_rate),
      transform.add_noise(eta, gamma, seed),
  )


def rmsprop(learning_rate: float,
            decay: float = 0.9,
            eps: float = 1e-8,
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


def sgd(learning_rate: float,
        momentum: float = 0.,
        nesterov: bool = False) -> GradientTransformation:
  return combine.chain(
      transform.trace(decay=momentum, nesterov=nesterov),
      transform.scale(-learning_rate),
  )
