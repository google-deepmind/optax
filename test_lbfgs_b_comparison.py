#!/usr/bin/env python3
"""
Complete comparison of Optax L-BFGS, Optax L-BFGS-B, and PyTorch L-BFGS
with both quadratic and Rosenbrock functions, warm-up runs, an    # === WARM-UP RUNS (JIT compilation) ===
    if verbose:
        print("Running warm-up iterations...")
    
    # Determine if this is a complex function that needs more warm-up
    test_result = objective_fn_jax(initial_params)
    is_complex_function = test_result > 10.0  # Heuristic: Rosenbrock starts high
    
    warm_up_iterations = 5 if not is_complex_function else 15
    warm_up_rounds = 3 if not is_complex_function else 5
    
    if verbose and is_complex_function:
        print(f"Detected complex function (initial value: {test_result:.2f}), using extended warm-up...")
    
    # Warm up Optax optimizers
    for _ in range(warm_up_rounds):
        optimize_optax_lbfgs(initial_params, objective_fn_jax, max_iter=warm_up_iterations)
        if bounds:
            optimize_optax_lbfgs_b(initial_params, objective_fn_jax, bounds=bounds, max_iter=warm_up_iterations)sis.
"""

import jax
import jax.numpy as jnp
import optax
import torch
import torch.optim as optim
import time
import numpy as np

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
    return 100 * (y - x**2)**2 + (1 - x)**2

def rosenbrock_torch(params):
    """Rosenbrock function for PyTorch"""
    x, y = params[0], params[1]
    return 100 * (y - x**2)**2 + (1 - x)**2

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
        return initial_params, float('inf'), 0
    
    params = initial_params
    step_count = 0
    
    for i in range(max_iter):
        value, grads = jax.value_and_grad(objective_fn)(params)
        
        if verbose and i % 20 == 0:
            print(f"    Step {i}: f = {value:.6f}, params = ({params[0]:.4f}, {params[1]:.4f})")
        
        updates, opt_state = optimizer.update(grads, opt_state, params,
                                            value=value, grad=grads,
                                            value_fn=objective_fn)
        params = optax.apply_updates(params, updates)
        step_count = i + 1
        
        # Check convergence
        if jnp.linalg.norm(grads) < 1e-5:
            break
    
    final_value = objective_fn(params)
    return params, final_value, step_count

def optimize_optax_lbfgs_b(initial_params, objective_fn, bounds=None, max_iter=1000, verbose=False):
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
            print(f"    Step {i}: f = {value:.6f}, params = ({params[0]:.4f}, {params[1]:.4f})")
        
        updates, opt_state = optimizer.update(grads, opt_state, params,
                                            value=value, grad=grads,
                                            value_fn=objective_fn)
        params = optax.apply_updates(params, updates)
        step_count = i + 1
        
        # Check convergence
        if jnp.linalg.norm(grads) < 1e-5:
            break
    
    final_value = objective_fn(params)
    return params, final_value, step_count

def optimize_pytorch(initial_params, objective_fn, bounds=None, max_iter=1000, verbose=False):
    """Optimize using PyTorch L-BFGS"""
    params_tensor = torch.tensor(initial_params, requires_grad=True, dtype=torch.float64)
    
    if bounds is not None:
        # PyTorch L-BFGS doesn't support bounds directly, so we'll use projection
        lower_bounds, upper_bounds = bounds
        lower_tensor = torch.tensor(lower_bounds, dtype=torch.float64)
        upper_tensor = torch.tensor(upper_bounds, dtype=torch.float64)
    
    optimizer = optim.LBFGS([params_tensor], max_iter=max_iter, tolerance_grad=1e-5,
                           tolerance_change=1e-12, history_size=10, line_search_fn='strong_wolfe')
    
    step_count = 0
    
    def closure():
        nonlocal step_count
        optimizer.zero_grad()
        loss = objective_fn(params_tensor)
        loss.backward()
        
        if verbose and step_count % 20 == 0:
            print(f"    Step {step_count}: f = {loss.item():.6f}, params = ({params_tensor[0].item():.4f}, {params_tensor[1].item():.4f})")
        
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
# COMPARISON FUNCTIONS
# ============================================================================

def run_single_comparison(name, initial_params, objective_fn_jax, objective_fn_torch, bounds=None, verbose=False):
    """Run comparison for a single test function"""
    print(f"\n{name.upper()} FUNCTION")
    print("-" * 60)
    print(f"Initial parameters: ({initial_params[0]:.2f}, {initial_params[1]:.2f})")
    if bounds:
        print(f"Bounds: x ∈ [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}], y ∈ [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}]")
    else:
        print("Bounds: None (unconstrained)")
    print()
    
    results = {}
    
    # === WARM-UP RUNS (JIT compilation) ===
    if verbose:
        print("Running warm-up iterations...")
    
    # Warm up Optax optimizers with more comprehensive runs
    for _ in range(2):
        # Warm up with same initial params and function to ensure proper JIT compilation
        optimize_optax_lbfgs(initial_params, objective_fn_jax, max_iter=20)
        if bounds:
            optimize_optax_lbfgs_b(initial_params, objective_fn_jax, bounds=bounds, max_iter=20)
        
    # Additional warm-up specifically for complex functions like Rosenbrock
    if 'rosenbrock' in objective_fn_jax.__name__.lower() or any('rosenbrock' in str(objective_fn_jax).lower() for _ in [True]):
        if verbose:
            print("Extra warm-up for complex function...")
        for _ in range(3):
            optimize_optax_lbfgs(initial_params, objective_fn_jax, max_iter=30)
            if bounds:
                optimize_optax_lbfgs_b(initial_params, objective_fn_jax, bounds=bounds, max_iter=30)
    
    # === ACTUAL COMPARISONS ===
    
    # 1. Optax L-BFGS (unconstrained)
    print("1. Optax L-BFGS (unconstrained):")
    start_time = time.time()
    params_lbfgs, value_lbfgs, steps_lbfgs = optimize_optax_lbfgs(
        initial_params, objective_fn_jax, verbose=verbose)
    time_lbfgs = time.time() - start_time
    print(f"   Result: ({params_lbfgs[0]:.4f}, {params_lbfgs[1]:.4f}), f = {value_lbfgs:.6f}")
    print(f"   Steps: {steps_lbfgs}, Time: {time_lbfgs:.4f}s")
    results['lbfgs'] = (params_lbfgs, value_lbfgs, steps_lbfgs, time_lbfgs)
    
    # 2. Optax L-BFGS-B (bounded, if bounds provided)
    if bounds:
        print("\n2. Optax L-BFGS-B (bounded):")
        start_time = time.time()
        params_lbfgs_b, value_lbfgs_b, steps_lbfgs_b = optimize_optax_lbfgs_b(
            initial_params, objective_fn_jax, bounds=bounds, verbose=verbose)
        time_lbfgs_b = time.time() - start_time
        print(f"   Result: ({params_lbfgs_b[0]:.4f}, {params_lbfgs_b[1]:.4f}), f = {value_lbfgs_b:.6f}")
        print(f"   Steps: {steps_lbfgs_b}, Time: {time_lbfgs_b:.4f}s")
        results['lbfgs_b'] = (params_lbfgs_b, value_lbfgs_b, steps_lbfgs_b, time_lbfgs_b)
    
    # 3. PyTorch L-BFGS
    print(f"\n{'3' if bounds else '2'}. PyTorch L-BFGS:")
    start_time = time.time()
    params_pytorch, value_pytorch, steps_pytorch = optimize_pytorch(
        initial_params, objective_fn_torch, bounds=bounds, verbose=verbose)
    time_pytorch = time.time() - start_time
    print(f"   Result: ({params_pytorch[0]:.4f}, {params_pytorch[1]:.4f}), f = {value_pytorch:.6f}")
    print(f"   Steps: {steps_pytorch}, Time: {time_pytorch:.4f}s")
    results['pytorch'] = (params_pytorch, value_pytorch, steps_pytorch, time_pytorch)
    
    # === SUMMARY ===
    print(f"\n{name.upper()} SUMMARY:")
    print("=" * 40)
    
    # Speed comparisons
    if bounds and 'lbfgs_b' in results:
        ratio_b_vs_pytorch = time_lbfgs_b / time_pytorch
        print(f"Optax L-BFGS-B vs PyTorch: {ratio_b_vs_pytorch:.2f}x")
    
    ratio_lbfgs_vs_pytorch = time_lbfgs / time_pytorch
    print(f"Optax L-BFGS vs PyTorch: {ratio_lbfgs_vs_pytorch:.2f}x")
    
    if bounds and 'lbfgs_b' in results:
        ratio_lbfgs_vs_b = time_lbfgs / time_lbfgs_b
        print(f"Optax L-BFGS vs L-BFGS-B: {ratio_lbfgs_vs_b:.2f}x")
    
    # Convergence comparison
    print("\nConvergence steps:")
    print(f"Optax L-BFGS: {steps_lbfgs}")
    if bounds and 'lbfgs_b' in results:
        print(f"Optax L-BFGS-B: {steps_lbfgs_b}")
    print(f"PyTorch L-BFGS: {steps_pytorch}")
    
    return results

def run_complete_comparison():
    """Run complete comparison with both test functions"""
    print("COMPLETE L-BFGS OPTIMIZER COMPARISON")
    print("=" * 80)
    print("Testing both quadratic and Rosenbrock functions")
    print("All times shown are after JIT warm-up (fair comparison)")
    print()
    
    # Test configurations
    configs = [
        {
            'name': 'quadratic',
            'initial_params': jnp.array([2.0, 2.0]),
            'objective_jax': quadratic_jax,
            'objective_torch': quadratic_torch,
            'bounds': (jnp.array([-3.0, -3.0]), jnp.array([3.0, 3.0]))
        },
        {
            'name': 'rosenbrock',
            'initial_params': jnp.array([-1.2, 1.0]),
            'objective_jax': rosenbrock_jax,
            'objective_torch': rosenbrock_torch,
            'bounds': (jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0]))
        }
    ]
    
    all_results = {}
    
    for config in configs:
        results = run_single_comparison(
            config['name'],
            config['initial_params'],
            config['objective_jax'],
            config['objective_torch'],
            config['bounds'],
            verbose=False
        )
        all_results[config['name']] = results
    
    # === OVERALL SUMMARY ===
    print("\n\nOVERALL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for func_name, results in all_results.items():
        print(f"\n{func_name.upper()}:")
        
        # Extract times
        lbfgs_time = results['lbfgs'][3]
        pytorch_time = results['pytorch'][3]
        
        if 'lbfgs_b' in results:
            lbfgs_b_time = results['lbfgs_b'][3]
            print(f"  L-BFGS-B: {lbfgs_b_time:.4f}s ({lbfgs_b_time/pytorch_time:.2f}x vs PyTorch)")
        
        print(f"  L-BFGS:   {lbfgs_time:.4f}s ({lbfgs_time/pytorch_time:.2f}x vs PyTorch)")
        print(f"  PyTorch:  {pytorch_time:.4f}s (baseline)")
    
    print("\nNOTE: All Optax times are after JIT compilation (warm-up).")
    print("First-run times would be significantly higher due to compilation overhead.")

if __name__ == "__main__":
    run_complete_comparison()
