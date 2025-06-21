# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple, Any
from optax._src import base
from optax import tree_utils


class _EmptyState(NamedTuple):
    pass


ScalarOrTree = Any  # PyTree of scalars or arrays


def _clip_to_bounds(params, lower_bounds, upper_bounds):
    """Clip parameters to box bounds using native tree operations."""
    if lower_bounds is None and upper_bounds is None:
        return params

    # Use tree_utils.tree_clip for efficient clipping
    return tree_utils.tree_clip(params, lower_bounds, upper_bounds)


def _clip_leaf(param, lower, upper):
    """Clip a single leaf to bounds."""
    if lower is None and upper is None:
        return param
    if lower is None:
        return jnp.minimum(param, upper)
    if upper is None:
        return jnp.maximum(param, lower)
    return jnp.clip(param, lower, upper)


def project_params_to_bounds(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree]
) -> base.GradientTransformation:
    """
    Projects parameters into the box constraints [lower_bounds, upper_bounds].

    Args:
        lower_bounds: PyTree of lower bounds (None for unbounded).
        upper_bounds: PyTree of upper bounds (None for unbounded).

    Returns:
        A GradientTransformation that on update ignores incoming updates
        and produces the delta to move params into the feasible region.
    """
    def init_fn(params):
        return _EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("project_params_to_bounds requires current params.")
        clipped = jax.tree_util.tree_map(_clip_leaf, params, lower_bounds, upper_bounds)
        deltas = jax.tree_util.tree_map(lambda c, p: c - p, clipped, params)
        return deltas, state

    return base.GradientTransformation(init_fn, update_fn)


def project_gradients_at_bounds(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree],
    tolerance: float = 1e-8
) -> base.GradientTransformation:
    """
    Zeroes gradients at active box constraints.

    For any parameter p at its lower bound (p <= lower+tol) and gradient > 0,
    or p at its upper bound (p >= upper-tol) and gradient < 0,
    sets gradient to 0 to prevent infeasible updates.
    """
    def init_fn(params):
        return _EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("project_gradients_at_bounds requires current params.")

        def proj_grad(grad, param, lower, upper):
            zero = jnp.zeros_like(grad)
            # Check lower bound constraint
            if lower is not None:
                at_lower = (param <= lower + tolerance) & (grad > 0)
            else:
                at_lower = jnp.zeros_like(grad, dtype=bool)

            # Check upper bound constraint
            if upper is not None:
                at_upper = (param >= upper - tolerance) & (grad < 0)
            else:
                at_upper = jnp.zeros_like(grad, dtype=bool)

            # Zero out gradients at active constraints
            return jnp.where(at_lower | at_upper, zero, grad)

        new_updates = jax.tree_util.tree_map(
            proj_grad, updates, params, lower_bounds, upper_bounds)
        return new_updates, state

    return base.GradientTransformation(init_fn, update_fn)


def final_clip_params(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree]
) -> base.GradientTransformation:
    """
    Clips parameters to the feasible region after applying updates.

    This transform takes incoming updates and current params, computes
    new_params = params + updates, clips new_params into [lower_bounds,
    upper_bounds], and returns the clipped delta.
    """
    def init_fn(params):
        return _EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("final_clip_params requires current params.")
        new_params = jax.tree_util.tree_map(
            lambda param, update: param + update, params, updates)
        clipped = jax.tree_util.tree_map(
            _clip_leaf, new_params, lower_bounds, upper_bounds)
        new_updates = jax.tree_util.tree_map(
            lambda clipped_param, param: clipped_param - param, clipped, params)
        return new_updates, state

    return base.GradientTransformation(init_fn, update_fn)
