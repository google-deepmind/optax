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
"""Smoothing functions."""

from typing import Union

import jax
import jax.numpy as jnp
from optax._src import utils


def smooth_labels(
    labels: jax.typing.ArrayLike,
    alpha: jax.typing.ArrayLike,
    *,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  """Apply label smoothing.

  Label smoothing is often used in combination with a cross-entropy loss.
  Smoothed labels favor small logit gaps, and it has been shown that this can
  provide better model calibration by preventing overconfident predictions.

  Args:
    labels: One hot labels to be smoothed.
    alpha: The smoothing factor.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    a smoothed version of the one hot input labels.

  References:
    Muller et al, `When does label smoothing help?
    <https://arxiv.org/abs/1906.02629>`_, 2019
  """
  utils.check_subdtype(labels, jnp.floating)
  if where is None:
    num_categories = jnp.size(labels, axis)
  else:
    num_categories = jnp.sum(where, axis, keepdims=True)
  return (1.0 - alpha) * labels + alpha / num_categories
