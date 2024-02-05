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
"""Wraps a GradientTransform into a Solver."""

from typing import Any, NamedTuple, Union

import jax
import optax
import optax.experimental.solver as optax_solver
import optax.experimental.utils as exp_utils


class GradientSolverState(NamedTuple):
  gt_state: optax.OptState = None


def gradient_solver(obj_fn, gradient_transform, obj_fun_has_aux=False):
  """Wraps a GradientTransform into a Solver."""

  def init(init_params: optax.Params) -> optax_solver.SolverState:
    init_gt_state = gradient_transform.init(init_params)
    init_opt_state = GradientSolverState(init_gt_state)
    return init_opt_state

  def step(
      params: optax.Params,
      state: optax_solver.SolverState,
      **extra_kwargs: dict[str, Any]
  ) -> tuple[
      Union[optax.Params, tuple[optax.Params, Any]], optax_solver.SolverState
  ]:
    obj_kwargs, gt_kwargs = exp_utils.split_kwargs(
        (obj_fn, gradient_transform.update), extra_kwargs
    )
    if obj_fun_has_aux:
      grad, aux = jax.grad(obj_fn, has_aux=obj_fun_has_aux)(
          params, **obj_kwargs
      )
    else:
      grad = jax.grad(obj_fn)(params, **obj_kwargs)
      aux = None
    update, gt_state = gradient_transform.update(
        grad, state.gt_state, params, **gt_kwargs
    )
    next_params = optax.apply_updates(params, update)
    next_state = GradientSolverState(gt_state)
    if obj_fun_has_aux:
      return (next_params, aux), next_state
    else:
      return next_params, next_state

  return optax_solver.Solver(init, step)
