import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple, Any
import optax

class _EmptyState(NamedTuple):
    pass

ScalarOrTree = Any  # PyTree of scalars or arrays


def _clip_leaf(p, l, u):
    """Clip a single leaf to bounds."""
    if l is None and u is None:
        return p
    if l is None:
        return jnp.minimum(p, u)
    if u is None:
        return jnp.maximum(p, l)
    return jnp.clip(p, l, u)


def project_params_to_bounds(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree]
) -> optax.GradientTransformation:
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

    return optax.GradientTransformation(init_fn, update_fn)


def project_gradients_at_bounds(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree],
    tolerance: float = 1e-8
) -> optax.GradientTransformation:
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
        def proj_grad(g, p, l, u):
            zero = jnp.zeros_like(g)
            # Check lower bound constraint
            if l is not None:
                at_lower = (p <= l + tolerance) & (g > 0)
            else:
                at_lower = jnp.zeros_like(g, dtype=bool)
            
            # Check upper bound constraint  
            if u is not None:
                at_upper = (p >= u - tolerance) & (g < 0)
            else:
                at_upper = jnp.zeros_like(g, dtype=bool)
                
            # Zero out gradients at active constraints
            return jnp.where(at_lower | at_upper, zero, g)
        new_updates = jax.tree_util.tree_map(proj_grad, updates, params, lower_bounds, upper_bounds)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def final_clip_params(
    lower_bounds: Optional[ScalarOrTree],
    upper_bounds: Optional[ScalarOrTree]
) -> optax.GradientTransformation:
    """
    Clips parameters to the feasible region after applying updates.

    This transform takes incoming updates and current params, computes new_params = params + updates,
    clips new_params into [lower_bounds, upper_bounds], and returns the clipped delta.
    """
    def init_fn(params):
        return _EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("final_clip_params requires current params.")
        new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        clipped = jax.tree_util.tree_map(_clip_leaf, new_params, lower_bounds, upper_bounds)
        new_updates = jax.tree_util.tree_map(lambda c, p: c - p, clipped, params)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)