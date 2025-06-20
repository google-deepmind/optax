#!/usr/bin/env python3

"""Test module to verify the integrated L-BFGS-B implementation."""

import jax
import jax.numpy as jnp
import optax
import numpy as np


def test_lbfgs_b_quadratic_unbounded():
    """Test L-BFGS-B on simple quadratic function without bounds."""

    def objective(x):
        return jnp.sum((x - 1.5) ** 2)

    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b()
    opt_state = solver.init(x0)

    x = x0
    for i in range(10):
        grad = jax.grad(objective)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x, value=objective(x), grad=grad, value_fn=objective
        )
        x = optax.apply_updates(x, updates)

        if objective(x) < 1e-6:
            break

    # Verify convergence to the correct minimum
    expected = jnp.array([1.5, 1.5])
    assert np.linalg.norm(x - expected) < 1e-3, f"Expected {expected}, got {x}"


def test_lbfgs_b_quadratic_bounded():
    """Test L-BFGS-B on simple quadratic function with bounds."""

    def objective(x):
        return jnp.sum((x - 1.5) ** 2)

    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b(
        lower_bounds=jnp.array([0.0, 0.0]), upper_bounds=jnp.array([1.0, 1.0])
    )
    opt_state = solver.init(x0)

    x = x0
    for i in range(10):
        grad = jax.grad(objective)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x, value=objective(x), grad=grad, value_fn=objective
        )
        x = optax.apply_updates(x, updates)

        if jnp.linalg.norm(grad) < 1e-6:
            break

    # Verify convergence to the constrained minimum
    expected = jnp.array([1.0, 1.0])  # Constrained to [1,1] instead of [1.5, 1.5]
    assert np.linalg.norm(x - expected) < 1e-3, f"Expected {expected}, got {x}"


def test_lbfgs_b_rosenbrock():
    """Test L-BFGS-B on Rosenbrock function."""

    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b()
    opt_state = solver.init(x0)

    x = x0
    for i in range(20):
        grad = jax.grad(rosenbrock)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x, value=rosenbrock(x), grad=grad, value_fn=rosenbrock
        )
        x = optax.apply_updates(x, updates)

        if rosenbrock(x) < 1e-6:
            break

    # Verify convergence to the Rosenbrock minimum
    expected = jnp.array([1.0, 1.0])
    assert np.linalg.norm(x - expected) < 0.15, f"Expected {expected}, got {x}"


def test_lbfgs_b_jit_compatibility():
    """Test L-BFGS-B JIT compatibility."""

    def objective(x):
        return jnp.sum((x - 1.5) ** 2)

    solver = optax.lbfgs_b()

    @jax.jit
    def step_fn(x, opt_state):
        grad = jax.grad(objective)(x)
        updates, new_opt_state = solver.update(
            grad, opt_state, x, value=objective(x), grad=grad, value_fn=objective
        )
        new_x = optax.apply_updates(x, updates)
        return new_x, new_opt_state

    x0 = jnp.array([0.0, 0.0])
    opt_state = solver.init(x0)

    x = x0
    for i in range(5):
        x, opt_state = step_fn(x, opt_state)

    # Verify that optimization made progress
    assert objective(x) < objective(x0)


if __name__ == "__main__":
    # Run tests when executed directly
    test_lbfgs_b_quadratic_unbounded()
    test_lbfgs_b_quadratic_bounded()
    test_lbfgs_b_rosenbrock()
    test_lbfgs_b_jit_compatibility()
    print("All tests passed!")
