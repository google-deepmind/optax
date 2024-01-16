# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Polyak SGD in Optax format."""

from typing import Optional

import jax
import jax.numpy as jnp
from optax import tree_utils
from optax._src import alias
from optax._src import base
from optax._src import combine

PolyakStepSGDState = base.EmptyState


def scale_by_polyak_step(
    f_star: float = 0.0,
    max_stepsize: float = 1.0,
    delta: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  """Scales the update by the Polyak step-size."""

  def init_fn(params: base.Params) -> PolyakStepSGDState:
    del params
    return PolyakStepSGDState()

  def update_fn(
      updates: base.Updates,
      state: PolyakStepSGDState,
      params: Optional[base.Params],
      *,
      value: Optional[jax.Array]) -> tuple[base.Updates, PolyakStepSGDState]:
    del params
    grad_sq_norm = tree_utils.tree_l2_norm(updates, squared=True)
    step = jnp.minimum(
        (value - f_star) / (grad_sq_norm + delta), max_stepsize
    ).item()
    new_updates = tree_utils.tree_scalar_mul(step, updates)
    return new_updates, state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def polyak_step_sgd(
    f_star=0.0, max_stepsize: float = 1.0, delta: float = 0.0
) -> base.GradientTransformationExtraArgs:
  # pylint: disable=line-too-long
  r"""SGD with Polyak step size.

  This solver implements the SGD with Polyak step size of (Loizou et al. 2021),
  which accepts the hyperparameters ``max_stepsize`` and ``delta`` and sets
  the current step-size as

  .. math::

    \min\left\{\frac{f(x) - f(x_\star)}{\|\nabla f(x)\|^2 + \text{delta}}, \text{max_stepsize}\right\}

  where :math:`f` is the current objective function and :math:`f(x_\star)` is

  Args:
    f_star: a lower bound on the objective function (defaults to 0). Corresponds
      to :math:`f(x_\star)` in the formula above.
    max_stepsize: a maximum step size to use (defaults to 1).
    delta: a value to add in the denominator of the update (defaults to 0).

  Returns:
    A `GradientTransformation` with init and update functions.

  .. warning::
      This method requires knowledge of an approximate value of the of the
      objective function minimum, passed through the ``fun_min`` argument.
      For models that interpolate the data, this can be set to 0 (default value).
      Failing to set an appropriate value for ``fun_min`` can lead to
      divergence or convergence to a suboptimal solution.

  References:
    Loizou, Nicolas and Vaswani, Sharan and Laradji, Issam Hadj and
    Lacoste-Julien, Simon.
    "Stochastic polyak step-size for sgd: An adaptive learning rate for fast
    convergence".
    International Conference on Artificial Intelligence and Statistics, 2021.
    https://arxiv.org/abs/2002.10542
  """
  # pylint: enable=line-too-long
  return combine.chain(
      alias.sgd(learning_rate=1.0),
      scale_by_polyak_step(
          max_stepsize=max_stepsize, f_star=f_star, delta=delta
      ),
  )
