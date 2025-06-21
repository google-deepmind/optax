#!/usr/bin/env python3
"""Test that sigmoid_focal_loss matches sigmoid_binary_cross_entropy when gamma=0."""

import jax
import jax.numpy as jnp
import numpy as np
from optax.losses import sigmoid_focal_loss, sigmoid_binary_cross_entropy


def test_gamma_zero_equivalence():
    """Test that focal loss with gamma=0 equals binary cross entropy."""
    # Test cases covering various scenarios
    test_cases = [
        # Normal range logits
        (jnp.array([0.5, -0.5, 1.0, -1.0]), jnp.array([1.0, 0.0, 1.0, 0.0])),
        # Extreme positive logits
        (jnp.array([50.0, 100.0]), jnp.array([1.0, 1.0])),
        # Extreme negative logits
        (jnp.array([-50.0, -100.0]), jnp.array([0.0, 0.0])),
        # Mixed extreme logits
        (jnp.array([50.0, -50.0, 100.0, -100.0]), jnp.array([1.0, 0.0, 1.0, 0.0])),
        # Continuous labels (label smoothing)
        (jnp.array([2.0, -2.0, 0.0]), jnp.array([0.9, 0.1, 0.5])),
        # Zero logits
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])),
    ]
    
    for i, (logits, labels) in enumerate(test_cases):
        print(f"\nTest case {i+1}: logits={logits}, labels={labels}")
        
        # Compute focal loss with gamma=0
        focal_loss = sigmoid_focal_loss(logits, labels, gamma=0.0)
        
        # Compute standard binary cross entropy
        bce_loss = sigmoid_binary_cross_entropy(logits, labels)
        
        print(f"Focal loss (γ=0): {focal_loss}")
        print(f"Binary CE:        {bce_loss}")
        print(f"Max absolute diff: {jnp.max(jnp.abs(focal_loss - bce_loss))}")
        
        # Check equivalence with reasonable tolerance
        np.testing.assert_allclose(
            focal_loss, bce_loss, 
            rtol=1e-6, atol=1e-8,
            err_msg=f"Test case {i+1} failed: focal loss != BCE when gamma=0"
        )
        print("✓ PASSED")
    
    print("\n" + "="*60)
    print("All tests passed! Focal loss with gamma=0 equals binary cross entropy.")


def test_with_alpha_weighting():
    """Test gamma=0 equivalence with alpha weighting."""
    print("\nTesting with alpha weighting...")
    
    logits = jnp.array([1.0, -1.0, 2.0, -2.0])
    labels = jnp.array([1.0, 0.0, 1.0, 0.0])
    alpha = 0.25
    
    # Focal loss with gamma=0 and alpha
    focal_loss = sigmoid_focal_loss(logits, labels, alpha=alpha, gamma=0.0)
    
    # Manual computation: alpha-weighted BCE
    bce_loss = sigmoid_binary_cross_entropy(logits, labels)
    alpha_weights = alpha * labels + (1 - alpha) * (1 - labels)
    weighted_bce = alpha_weights * bce_loss
    
    print(f"Focal loss (γ=0, α={alpha}): {focal_loss}")
    print(f"Weighted BCE:                 {weighted_bce}")
    print(f"Max absolute diff: {jnp.max(jnp.abs(focal_loss - weighted_bce))}")
    
    np.testing.assert_allclose(
        focal_loss, weighted_bce,
        rtol=1e-6, atol=1e-8,
        err_msg="Alpha-weighted focal loss != alpha-weighted BCE when gamma=0"
    )
    print("✓ PASSED")


if __name__ == "__main__":
    test_gamma_zero_equivalence()
    test_with_alpha_weighting()
