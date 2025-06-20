#!/usr/bin/env python3

"""Test script to verify the integrated L-BFGS-B implementation."""

import jax
import jax.numpy as jnp
import optax
import numpy as np

def test_simple_quadratic():
    """Test simple quadratic function."""
    print("Testing simple quadratic function...")
    
    def objective(x):
        return jnp.sum((x - 1.5)**2)
    
    # Test unbounded case
    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b()
    opt_state = solver.init(x0)
    
    x = x0
    for i in range(10):
        grad = jax.grad(objective)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x, 
            value=objective(x), grad=grad, value_fn=objective
        )
        x = optax.apply_updates(x, updates)
        print(f"Step {i}: x={x}, f={objective(x)}")
        
        if objective(x) < 1e-6:
            break
    
    print(f"Final x: {x}, target: [1.5, 1.5]")
    print(f"Error: {np.linalg.norm(x - 1.5)}\n")
    
    # Test bounded case
    print("Testing bounded quadratic...")
    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b(
        lower_bounds=jnp.array([0.0, 0.0]),
        upper_bounds=jnp.array([1.0, 1.0])
    )
    opt_state = solver.init(x0)
    
    x = x0
    for i in range(10):
        grad = jax.grad(objective)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x,
            value=objective(x), grad=grad, value_fn=objective
        )
        x = optax.apply_updates(x, updates)
        print(f"Step {i}: x={x}, f={objective(x)}")
        
        if jnp.linalg.norm(grad) < 1e-6:
            break
    
    print(f"Final x: {x}, target: [1.0, 1.0] (bounded)")
    print(f"Error: {np.linalg.norm(x - 1.0)}\n")


def test_rosenbrock():
    """Test Rosenbrock function."""
    print("Testing Rosenbrock function...")
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b()
    opt_state = solver.init(x0)
    
    x = x0
    for i in range(20):
        grad = jax.grad(rosenbrock)(x)
        updates, opt_state = solver.update(
            grad, opt_state, x,
            value=rosenbrock(x), grad=grad, value_fn=rosenbrock
        )
        x = optax.apply_updates(x, updates)
        
        if i % 5 == 0:
            print(f"Step {i}: x={x}, f={rosenbrock(x)}")
            
        if rosenbrock(x) < 1e-6:
            break
    
    print(f"Final x: {x}, target: [1.0, 1.0]")
    print(f"Error: {np.linalg.norm(x - 1.0)}\n")


def test_jit_compatibility():
    """Test JIT compatibility."""
    print("Testing JIT compatibility...")
    
    def objective(x):
        return jnp.sum((x - 1.5)**2)
    
    @jax.jit
    def step_fn(x, opt_state):
        grad = jax.grad(objective)(x)
        updates, new_opt_state = solver.update(
            grad, opt_state, x,
            value=objective(x), grad=grad, value_fn=objective
        )
        new_x = optax.apply_updates(x, updates)
        return new_x, new_opt_state
    
    x0 = jnp.array([0.0, 0.0])
    solver = optax.lbfgs_b()
    opt_state = solver.init(x0)
    
    x = x0
    for i in range(5):
        x, opt_state = step_fn(x, opt_state)
        print(f"JIT Step {i}: x={x}, f={objective(x)}")
    
    print(f"JIT Final x: {x}")
    print("JIT compatibility: PASSED\n")


if __name__ == "__main__":
    print("Testing integrated L-BFGS-B implementation...\n")
    
    test_simple_quadratic()
    test_rosenbrock()
    test_jit_compatibility()
    
    print("All tests completed!")
