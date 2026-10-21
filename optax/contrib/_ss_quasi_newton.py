# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Self-Scaled Quasi-Newton optimizers (SSBFGS and SSBroyden).

References:
  Urbán et al., 'Unveiling the optimization process of physics informed neural
  networks: How accurate and competitive can PINNs be?', Journal of
  Computational Physics, 2025.
"""

from typing import Any, NamedTuple, Optional
import jax
from jax import flatten_util
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import linesearch
from optax._src import transform


class ScaleBySSQuasiNewtonState(NamedTuple):
  count: jax.Array
  inv_hessian: jax.Array
  prev_flat_params: jax.Array
  prev_flat_grad: jax.Array


def scale_by_ss_quasi_newton(
    phi: float = 0.0,
    curvature_threshold: float = 1e-8,
) -> base.GradientTransformation:

  def init_fn(params: base.Params) -> ScaleBySSQuasiNewtonState:
    flat_params, _ = flatten_util.ravel_pytree(params)
    dim = flat_params.shape[0]
    return ScaleBySSQuasiNewtonState(
        count=jnp.zeros([], dtype=jnp.int32),
        inv_hessian=jnp.eye(dim, dtype=flat_params.dtype),
        prev_flat_params=flat_params,
        prev_flat_grad=jnp.zeros_like(flat_params),
    )

  def update_fn(
      updates: base.Updates,
      state: ScaleBySSQuasiNewtonState,
      params: Optional[base.Params] = None,
      **extra_args: Any,
  ) -> tuple[base.Updates, ScaleBySSQuasiNewtonState]:
    del extra_args
    if params is None:
      raise ValueError("Quasi-Newton methods require current params.")

    flat_grad, unravel_fn = flatten_util.ravel_pytree(updates)
    flat_params, _ = flatten_util.ravel_pytree(params)

    def _first_step():
      # Initial iteration: use initial H (identity) to compute direction
      direction = -jnp.matmul(state.inv_hessian, flat_grad)
      new_state = ScaleBySSQuasiNewtonState(
          count=state.count + 1,
          inv_hessian=state.inv_hessian,
          prev_flat_params=flat_params,
          prev_flat_grad=flat_grad,
      )
      return direction, new_state

    def _later_step():
      s = flat_params - state.prev_flat_params
      y = flat_grad - state.prev_flat_grad

      s_dot_y = jnp.dot(s, y)
      Hy = jnp.matmul(state.inv_hessian, y)
      yHy = jnp.dot(y, Hy)

      def _update_hessian():
        # Oren-Luenberger self-scaling factor
        gamma = s_dot_y / jnp.maximum(yHy, curvature_threshold)
        H0 = gamma * state.inv_hessian

        H0_y = gamma * Hy
        y_H0_y = gamma * yHy

        # Rank-2 update components
        term1 = jnp.outer(H0_y, H0_y) / jnp.maximum(y_H0_y, curvature_threshold)
        term2 = jnp.outer(s, s) / jnp.maximum(s_dot_y, curvature_threshold)

        v = (s / jnp.maximum(s_dot_y, curvature_threshold)) - (
            H0_y / jnp.maximum(y_H0_y, curvature_threshold)
        )
        term3 = (1.0 - phi) * y_H0_y * jnp.outer(v, v)

        H_next = H0 - term1 + term2 + term3
        return H_next

      # Only update H if the curvature condition s^T y > threshold holds
      h_next = jax.lax.cond(
          s_dot_y > curvature_threshold,
          _update_hessian,
          lambda: state.inv_hessian,
      )

      direction = -jnp.matmul(h_next, flat_grad)
      new_state = ScaleBySSQuasiNewtonState(
          count=state.count + 1,
          inv_hessian=h_next,
          prev_flat_params=flat_params,
          prev_flat_grad=flat_grad,
      )
      return direction, new_state

    flat_direction, new_state = jax.lax.cond(
        state.count == 0,
        _first_step,
        _later_step,
    )

    out_updates = unravel_fn(flat_direction)
    return out_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def ssbfgs(
    learning_rate: float = 1.0,
    linesearch_fn: Optional[base.GradientTransformationExtraArgs] = None,
    curvature_threshold: float = 1e-8,
) -> base.GradientTransformation:

  if linesearch_fn is None:
    linesearch_fn = linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=20, initial_guess_strategy="one"
    )

  return combine.chain(
      scale_by_ss_quasi_newton(
          phi=0.0, curvature_threshold=curvature_threshold
      ),
      linesearch_fn,
      transform.scale(learning_rate),
  )


def ssbroyden(
    phi: float = 0.5,
    learning_rate: float = 1.0,
    linesearch_fn: Optional[base.GradientTransformationExtraArgs] = None,
    curvature_threshold: float = 1e-8,
) -> base.GradientTransformation:

  if linesearch_fn is None:
    linesearch_fn = linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=20, initial_guess_strategy="one"
    )

  return combine.chain(
      scale_by_ss_quasi_newton(
          phi=phi, curvature_threshold=curvature_threshold
      ),
      linesearch_fn,
      transform.scale(learning_rate),
  )
