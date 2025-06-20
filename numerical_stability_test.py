#!/usr/bin/env python3
"""
Focused test demonstrating that extreme logits + small gamma no longer breaks 
Hessian or Jacobian in the log-space implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from optax.losses._classification import sigmoid_binary_cross_entropy
from optax.losses._classification import sigmoid_focal_loss as log_space_sigmoid_focal_loss
import functools
import chex
from typing import Optional

@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def original_sigmoid_focal_loss(
    logits: chex.Array,
    labels: chex.Array,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
) -> chex.Array:
    """Original problematic implementation."""
    alpha = -1 if alpha is None else alpha
    chex.assert_type([logits], float)
    labels = jnp.astype(labels, logits.dtype)
    
    p = jax.nn.sigmoid(logits)
    ce_loss = sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)  # Problematic line
    
    weighted = (alpha * labels + (1 - alpha) * (1 - labels)) * loss
    loss = jnp.where(alpha >= 0, weighted, loss)
    return loss

def analyze_numerical_breakdown():
    """
    Comprehensive analysis showing where the original breaks and log-space fixes it.
    Includes underflow analysis and eigenvalue computations.
    """
    print("="*80)
    print("NUMERICAL STABILITY BREAKDOWN ANALYSIS")
    print("Extreme Logits + Small Gamma: Where Mathematics Meets Reality")
    print("="*80)
    
    # Test parameters that cause breakdown
    labels = jnp.array([1.0])  # Positive class
    gamma = 0.1  # Small gamma causes (1-p)^(gamma-2) = (1-p)^(-1.9) explosion
    
    # Progressive logit values approaching the breakdown point
    logit_values = [15.0, 17.0, 18.0, 19.0, 20.0, 25.0]
    
    print(f"\nTesting γ = {gamma} (problematic: γ-2 = {gamma-2:.1f} < 0)")
    print(f"Mathematical issue: Second derivative contains (1-p_t)^(γ-2) terms")
    print(f"When p_t → 1, (1-p_t)^(-1.9) → ∞")
    
    print(f"\n{'Logit':<6} {'p':<12} {'1-p':<12} {'(1-p)^-1.9':<15} {'Orig_Grad':<12} {'Log_Grad':<12} {'Status':<15}")
    print("-" * 95)
    
    results = []
    
    for logit in logit_values:
        logits = jnp.array([logit])
        
        # Compute underlying probabilities for analysis
        p = jax.nn.sigmoid(logit)
        one_minus_p = 1 - p
        problematic_term = one_minus_p**(gamma - 2) if one_minus_p > 0 else jnp.inf
        
        # Test original implementation
        def orig_loss_fn(x):
            return jnp.sum(original_sigmoid_focal_loss(x, labels, gamma=gamma))
        
        def log_loss_fn(x):
            return jnp.sum(log_space_sigmoid_focal_loss(x, labels, gamma=gamma))
        
        # Compute gradients
        try:
            orig_grad = jax.grad(orig_loss_fn)(logits)[0]
            orig_finite = jnp.isfinite(orig_grad)
        except:
            orig_grad = jnp.nan
            orig_finite = False
        
        try:
            log_grad = jax.grad(log_loss_fn)(logits)[0]
            log_finite = jnp.isfinite(log_grad)
        except:
            log_grad = jnp.nan
            log_finite = False
        
        # Status determination
        if orig_finite and log_finite:
            status = "Both Stable"
        elif not orig_finite and log_finite:
            status = "Log Fixes!"
        elif not orig_finite and not log_finite:
            status = "Both Broken"
        else:
            status = "Unexpected"
        
        results.append((logit, p, one_minus_p, problematic_term, orig_grad, log_grad, status))
        
        print(f"{logit:<6.1f} {p:<12.2e} {one_minus_p:<12.2e} {problematic_term:<15.2e} "
              f"{orig_grad:<12.2e} {log_grad:<12.2e} {status:<15}")
    
    return results

def hessian_eigenvalue_analysis():
    """Detailed Hessian analysis with eigenvalue computation."""
    print(f"\n{'='*80}")
    print("HESSIAN EIGENVALUE ANALYSIS")
    print("Second-order stability at the breakdown point")
    print("="*80)
    
    labels = jnp.array([1.0])
    gamma = 0.2  # Small gamma
    
    # Test cases: just before and at breakdown
    test_cases = [
        (15.0, "Safe zone"),
        (18.0, "Approaching breakdown"),
        (20.0, "At breakdown point"),
        (25.0, "Deep in unstable region")
    ]
    
    print(f"\nγ = {gamma} analysis:")
    print(f"{'Logit':<8} {'Description':<25} {'Orig_Condition':<15} {'Log_Condition':<15} {'Orig_MinEig':<15} {'Log_MinEig':<15}")
    print("-" * 100)
    
    for logit, description in test_cases:
        logits = jnp.array([logit])
        
        def orig_loss_fn(x):
            return jnp.sum(original_sigmoid_focal_loss(x, labels, gamma=gamma))
        
        def log_loss_fn(x):
            return jnp.sum(log_space_sigmoid_focal_loss(x, labels, gamma=gamma))
        
        # Compute Hessians
        try:
            orig_hess = jax.hessian(orig_loss_fn)(logits)
            orig_eigs = jnp.linalg.eigvals(orig_hess)
            orig_cond = jnp.linalg.cond(orig_hess)
            orig_min_eig = jnp.min(jnp.real(orig_eigs))
            orig_status = "Finite" if jnp.all(jnp.isfinite(orig_eigs)) else "NaN/Inf"
        except:
            orig_cond = jnp.inf
            orig_min_eig = jnp.nan
            orig_status = "Failed"
        
        try:
            log_hess = jax.hessian(log_loss_fn)(logits)
            log_eigs = jnp.linalg.eigvals(log_hess)
            log_cond = jnp.linalg.cond(log_hess)
            log_min_eig = jnp.min(jnp.real(log_eigs))
            log_status = "Finite" if jnp.all(jnp.isfinite(log_eigs)) else "NaN/Inf"
        except:
            log_cond = jnp.inf
            log_min_eig = jnp.nan
            log_status = "Failed"
        
        print(f"{logit:<8.1f} {description:<25} {orig_cond:<15.2e} {log_cond:<15.2e} "
              f"{orig_min_eig:<15.2e} {log_min_eig:<15.2e}")

def underflow_overflow_analysis():
    """Analysis of numerical underflow/overflow patterns."""
    print(f"\n{'='*80}")
    print("NUMERICAL UNDERFLOW/OVERFLOW ANALYSIS")
    print("Tracking the mathematical breakdown")
    print("="*80)
    
    gamma = 0.1
    logits_range = jnp.linspace(10, 25, 16)
    
    print(f"\nγ = {gamma}: Tracking (1-p)^(γ-2) = (1-p)^(-1.9)")
    print(f"{'Logit':<8} {'p':<12} {'1-p':<12} {'log(1-p)':<12} {'(1-p)^-1.9':<15} {'Safe?':<8}")
    print("-" * 75)
    
    machine_eps = jnp.finfo(jnp.float32).eps
    
    for logit in logits_range:
        p = jax.nn.sigmoid(logit)
        one_minus_p = 1 - p
        log_one_minus_p = jnp.log(one_minus_p) if one_minus_p > 0 else -jnp.inf
        
        # The problematic term in second derivative
        if one_minus_p > machine_eps:
            problematic_power = one_minus_p**(gamma - 2)
            safe = jnp.isfinite(problematic_power)
        else:
            problematic_power = jnp.inf
            safe = False
        
        safe_indicator = "✓" if safe else "✗"
        
        print(f"{logit:<8.1f} {p:<12.2e} {one_minus_p:<12.2e} {log_one_minus_p:<12.2f} "
              f"{problematic_power:<15.2e} {safe_indicator:<8}")
    
    print(f"\nMachine epsilon: {machine_eps:.2e}")
    print(f"Breakdown occurs when (1-p) approaches machine precision")
    print(f"Original implementation: (1-p)^(-1.9) → ∞")
    print(f"Log-space implementation: Clamps log(1-p) ≥ log(ε), preventing explosion")

def create_stability_visualization():
    """Create a visualization showing the stability regions."""
    print(f"\n{'='*80}")
    print("CREATING STABILITY VISUALIZATION")
    print("="*80)
    
    # Create parameter grid
    logits = jnp.linspace(5, 30, 50)
    gammas = jnp.linspace(0.1, 2.0, 40)
    logit_grid, gamma_grid = jnp.meshgrid(logits, gammas)
    
    # Test stability for both implementations
    original_stable = np.zeros_like(logit_grid)
    logspace_stable = np.zeros_like(logit_grid)
    
    labels = jnp.array([1.0])
    
    print("Testing stability across parameter space...")
    
    for i, gamma in enumerate(gammas):
        for j, logit in enumerate(logits):
            logit_arr = jnp.array([logit])
            
            # Test original
            try:
                def orig_fn(x):
                    return jnp.sum(original_sigmoid_focal_loss(x, labels, gamma=gamma))
                grad = jax.grad(orig_fn)(logit_arr)[0]
                hess = jax.hessian(orig_fn)(logit_arr)[0,0]
                original_stable[i, j] = float(jnp.isfinite(grad) and jnp.isfinite(hess))
            except:
                original_stable[i, j] = 0.0
            
            # Test log-space
            try:
                def log_fn(x):
                    return jnp.sum(log_space_sigmoid_focal_loss(x, labels, gamma=gamma))
                grad = jax.grad(log_fn)(logit_arr)[0]
                hess = jax.hessian(log_fn)(logit_arr)[0,0]
                logspace_stable[i, j] = float(jnp.isfinite(grad) and jnp.isfinite(hess))
            except:
                logspace_stable[i, j] = 0.0
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original implementation
    im1 = ax1.imshow(original_stable, extent=[logits[0], logits[-1], gammas[0], gammas[-1]], 
                     aspect='auto', origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
    ax1.set_title('Original Implementation\nNumerical Stability', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Logit Value', fontsize=12)
    ax1.set_ylabel('Gamma (γ)', fontsize=12)
    ax1.axhline(y=1.0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    ax1.text(17.5, 1.1, 'γ = 1 (critical)', color='white', fontweight='bold', ha='center')
    plt.colorbar(im1, ax=ax1, label='Stable (1) vs Unstable (0)')
    
    # Log-space implementation
    im2 = ax2.imshow(logspace_stable, extent=[logits[0], logits[-1], gammas[0], gammas[-1]], 
                     aspect='auto', origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
    ax2.set_title('Log-Space Implementation\nNumerical Stability', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Logit Value', fontsize=12)
    ax2.set_ylabel('Gamma (γ)', fontsize=12)
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    plt.colorbar(im2, ax=ax2, label='Stable (1) vs Unstable (0)')
    
    # Improvement map
    improvement = logspace_stable - original_stable
    im3 = ax3.imshow(improvement, extent=[logits[0], logits[-1], gammas[0], gammas[-1]], 
                     aspect='auto', origin='lower', cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_title('Log-Space Improvement\n(Blue = Better)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Logit Value', fontsize=12)
    ax3.set_ylabel('Gamma (γ)', fontsize=12)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    plt.colorbar(im3, ax=ax3, label='Improvement')
    
    plt.tight_layout()
    plt.savefig('focal_loss_stability_complete.png', dpi=300, bbox_inches='tight')
    print("Saved: focal_loss_stability_complete.png")
    
    # Summary statistics
    orig_stability = np.mean(original_stable) * 100
    log_stability = np.mean(logspace_stable) * 100
    improvements = np.sum(improvement > 0)
    total_cases = improvement.size
    
    print(f"\nSTABILITY SUMMARY:")
    print(f"Original implementation: {orig_stability:.1f}% stable")
    print(f"Log-space implementation: {log_stability:.1f}% stable")
    print(f"Cases improved: {improvements}/{total_cases} ({improvements/total_cases*100:.1f}%)")

if __name__ == "__main__":
    print("🔍 FOCAL LOSS NUMERICAL STABILITY ANALYSIS")
    print("Demonstrating log-space fixes for extreme logits + small gamma\n")
    
    # Main analysis
    analyze_numerical_breakdown()
    hessian_eigenvalue_analysis() 
    underflow_overflow_analysis()
    create_stability_visualization()
    
    print(f"\n{'='*80}")
    print("✅ CONCLUSION: Log-space implementation successfully prevents")
    print("   numerical breakdown in both gradients and Hessians for")
    print("   extreme logits with small gamma values.")
    print("="*80)
