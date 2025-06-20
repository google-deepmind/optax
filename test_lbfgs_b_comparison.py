#!/usr/bin/env python3
"""
Test module comparing Optax L-BFGS, Optax L-BFGS-B, and PyTorch L-BFGS
with both quadratic and Rosenbrock functions, including performance benchmarks.
"""

import jax
import jax.numpy as jnp
import optax
import torch
import torch.optim as optim
import time
import pytest


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def quadratic_jax(params):
    """Simple quadratic function for JAX: f(x,y) = x^2 + y^2"""
    x, y = params
    return x**2 + y**2


def quadratic_torch(params):
    """Simple quadratic function for PyTorch"""
    x, y = params[0], params[1]
    return x**2 + y**2


def rosenbrock_jax(params):
    """Rosenbrock function for JAX"""
    x, y = params
    return 100 * (y - x**2) ** 2 + (1 - x) ** 2


def rosenbrock_torch(params):
    """Rosenbrock function for PyTorch"""
    x, y = params[0], params[1]
    return 100 * (y - x**2) ** 2 + (1 - x) ** 2


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================


def optimize_optax_lbfgs(initial_params, objective_fn, max_iter=1000, verbose=False):
    """Optimize using standard Optax L-BFGS (unconstrained)"""
    try:
        optimizer = optax.lbfgs()
        opt_state = optimizer.init(initial_params)
    except AttributeError:
        # Fallback - skip this test if lbfgs not available
        return initial_params, float("inf"), 0

    params = initial_params
    step_count = 0

    for i in range(max_iter):
        value, grads = jax.value_and_grad(objective_fn)(params)

        if verbose and i % 20 == 0:
            print(
                f"    Step {i}: f = {value:.6f}, params = ({params[0]:.4f}, {params[1]:.4f})"
            )

        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=value, grad=grads, value_fn=objective_fn
        )
        params = optax.apply_updates(params, updates)
        step_count = i + 1

        # Check convergence
        if jnp.linalg.norm(grads) < 1e-5:
            break

    final_value = objective_fn(params)
    return params, final_value, step_count


def optimize_optax_lbfgs_b(
    initial_params, objective_fn, bounds=None, max_iter=1000, verbose=False
):
    """Optimize using Optax L-BFGS-B (constrained)"""
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        # Use the built-in bounded L-BFGS
        optimizer = optax.lbfgs_b(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    else:
        optimizer = optax.lbfgs()

    opt_state = optimizer.init(initial_params)
    params = initial_params
    step_count = 0

    for i in range(max_iter):
        value, grads = jax.value_and_grad(objective_fn)(params)

        if verbose and i % 20 == 0:
            print(
                f"    Step {i}: f = {value:.6f}, params = ({params[0]:.4f}, {params[1]:.4f})"
            )

        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=value, grad=grads, value_fn=objective_fn
        )
        params = optax.apply_updates(params, updates)
        step_count = i + 1

        # Check convergence
        if jnp.linalg.norm(grads) < 1e-5:
            break

    final_value = objective_fn(params)
    return params, final_value, step_count


def optimize_pytorch(
    initial_params, objective_fn, bounds=None, max_iter=1000, verbose=False
):
    """Optimize using PyTorch L-BFGS"""
    params_tensor = torch.tensor(
        initial_params, requires_grad=True, dtype=torch.float64
    )

    if bounds is not None:
        # PyTorch L-BFGS doesn't support bounds directly, so we'll use projection
        lower_bounds, upper_bounds = bounds
        lower_tensor = torch.tensor(lower_bounds, dtype=torch.float64)
        upper_tensor = torch.tensor(upper_bounds, dtype=torch.float64)

    optimizer = optim.LBFGS(
        [params_tensor],
        max_iter=max_iter,
        tolerance_grad=1e-5,
        tolerance_change=1e-12,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    step_count = 0

    def closure():
        nonlocal step_count
        optimizer.zero_grad()
        loss = objective_fn(params_tensor)
        loss.backward()

        if verbose and step_count % 20 == 0:
            print(
                f"    Step {step_count}: f = {loss.item():.6f}, params = ({params_tensor[0].item():.4f}, {params_tensor[1].item():.4f})"
            )

        step_count += 1
        return loss

    optimizer.step(closure)

    # Apply bounds if specified
    if bounds is not None:
        with torch.no_grad():
            params_tensor.clamp_(lower_tensor, upper_tensor)

    final_params = params_tensor.detach().numpy()
    final_value = objective_fn(params_tensor).item()

    return final_params, final_value, step_count


# ============================================================================
# PYTEST TEST FUNCTIONS
# ============================================================================


def test_lbfgs_b_vs_lbfgs_quadratic():
    """Test that L-BFGS-B converges similarly to L-BFGS on quadratic function."""
    initial_params = jnp.array([2.0, 2.0])

    # Optimize with L-BFGS
    params_lbfgs, value_lbfgs, steps_lbfgs = optimize_optax_lbfgs(
        initial_params, quadratic_jax, max_iter=50
    )

    # Optimize with L-BFGS-B (no bounds)
    params_lbfgs_b, value_lbfgs_b, steps_lbfgs_b = optimize_optax_lbfgs_b(
        initial_params, quadratic_jax, max_iter=50
    )

    # Both should converge to [0,0]
    assert jnp.allclose(params_lbfgs, jnp.array([0.0, 0.0]), atol=1e-4)
    assert jnp.allclose(params_lbfgs_b, jnp.array([0.0, 0.0]), atol=1e-4)
    assert value_lbfgs < 1e-6
    assert value_lbfgs_b < 1e-6


def test_lbfgs_b_bounded_optimization():
    """Test L-BFGS-B with bounds on quadratic function."""
    initial_params = jnp.array([2.0, 2.0])
    bounds = (jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))

    params, value, steps = optimize_optax_lbfgs_b(
        initial_params, quadratic_jax, bounds=bounds, max_iter=50
    )

    # Should converge to [0,0] which is within bounds
    assert jnp.allclose(params, jnp.array([0.0, 0.0]), atol=1e-4)
    assert value < 1e-6

    # Verify bounds are respected
    lower_bounds, upper_bounds = bounds
    assert jnp.all(params >= lower_bounds - 1e-6)
    assert jnp.all(params <= upper_bounds + 1e-6)


def test_lbfgs_b_constrained_optimization():
    """Test L-BFGS-B with tight bounds that constrain the solution."""

    def objective(x):
        return jnp.sum((x - 2.0) ** 2)  # Minimum at [2,2]

    initial_params = jnp.array([0.0, 0.0])
    bounds = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))  # Constrain to [0,1]x[0,1]

    params, value, steps = optimize_optax_lbfgs_b(
        initial_params, objective, bounds=bounds, max_iter=50
    )

    # Should converge to [1,1] (constrained minimum)
    assert jnp.allclose(params, jnp.array([1.0, 1.0]), atol=1e-4)

    # Verify bounds are respected
    lower_bounds, upper_bounds = bounds
    assert jnp.all(params >= lower_bounds - 1e-6)
    assert jnp.all(params <= upper_bounds + 1e-6)


def test_lbfgs_b_rosenbrock():
    """Test L-BFGS-B on Rosenbrock function."""
    initial_params = jnp.array([-1.2, 1.0])

    params, value, steps = optimize_optax_lbfgs_b(
        initial_params, rosenbrock_jax, max_iter=100
    )

    # Should converge to [1,1] (Rosenbrock minimum)
    assert jnp.allclose(params, jnp.array([1.0, 1.0]), atol=1e-2)
    assert value < 1e-4


@pytest.mark.parametrize(
    "objective_fn,initial_params,expected_min",
    [
        (quadratic_jax, jnp.array([2.0, 2.0]), jnp.array([0.0, 0.0])),
        (rosenbrock_jax, jnp.array([-1.2, 1.0]), jnp.array([1.0, 1.0])),
    ],
)
def test_lbfgs_b_optimization_functions(objective_fn, initial_params, expected_min):
    """Parametrized test for different objective functions."""
    params, value, steps = optimize_optax_lbfgs_b(
        initial_params, objective_fn, max_iter=100
    )

    tolerance = 1e-2 if "rosenbrock" in objective_fn.__name__ else 1e-4
    assert jnp.allclose(params, expected_min, atol=tolerance)


# ============================================================================
# PERFORMANCE COMPARISON FUNCTIONS (for manual running)
# ============================================================================


def run_performance_comparison():
    """Run performance comparison between optimizers (not a pytest test)."""
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    configs = [
        {
            "name": "quadratic",
            "initial_params": jnp.array([2.0, 2.0]),
            "objective_jax": quadratic_jax,
            "objective_torch": quadratic_torch,
            "bounds": (jnp.array([-3.0, -3.0]), jnp.array([3.0, 3.0])),
        },
        {
            "name": "rosenbrock",
            "initial_params": jnp.array([-1.2, 1.0]),
            "objective_jax": rosenbrock_jax,
            "objective_torch": rosenbrock_torch,
            "bounds": (jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0])),
        },
    ]

    for config in configs:
        print(f"\n{config['name'].upper()} FUNCTION")
        print("-" * 30)

        # Warm up JIT compilation
        for _ in range(3):
            optimize_optax_lbfgs(
                config["initial_params"], config["objective_jax"], max_iter=10
            )
            optimize_optax_lbfgs_b(
                config["initial_params"],
                config["objective_jax"],
                bounds=config["bounds"],
                max_iter=10,
            )

        # Actual timing
        start_time = time.time()
        params_lbfgs, value_lbfgs, steps_lbfgs = optimize_optax_lbfgs(
            config["initial_params"], config["objective_jax"]
        )
        time_lbfgs = time.time() - start_time

        start_time = time.time()
        params_lbfgs_b, value_lbfgs_b, steps_lbfgs_b = optimize_optax_lbfgs_b(
            config["initial_params"], config["objective_jax"], bounds=config["bounds"]
        )
        time_lbfgs_b = time.time() - start_time

        start_time = time.time()
        params_pytorch, value_pytorch, steps_pytorch = optimize_pytorch(
            config["initial_params"], config["objective_torch"], bounds=config["bounds"]
        )
        time_pytorch = time.time() - start_time

        print(
            f"L-BFGS:     {time_lbfgs:.4f}s, {steps_lbfgs} steps, f={value_lbfgs:.6f}"
        )
        print(
            f"L-BFGS-B:   {time_lbfgs_b:.4f}s, {steps_lbfgs_b} steps, f={value_lbfgs_b:.6f}"
        )
        print(
            f"PyTorch:    {time_pytorch:.4f}s, {steps_pytorch} steps, f={value_pytorch:.6f}"
        )


if __name__ == "__main__":
    run_performance_comparison()
