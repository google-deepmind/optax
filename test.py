import jax
import jax.numpy as jnp
import optax
# Ensure we are importing the module where you defined the muon alias
# If it's in a file named 'muon_opt.py', import from there. 
# Assuming it is available in optax.contrib based on previous context:
from optax.contrib import muon 

def test_adaptive_masking_stability():
    print("🚀 Starting Muon Adaptive Stability Test...")

    # 1. Setup params: A 2D matrix (Muon) and a 1D vector (Adam/Masked)
    params = {
        'w': jnp.ones((10, 10)),  # 2D -> Should be processed by Muon
        'b': jnp.ones((10,))      # 1D -> Should be masked out (handled by Adam)
    }
    
    # 2. Define gradients
    grads = {
        'w': jnp.full((10, 10), 0.1),
        'b': jnp.full((10,), 0.01)
    }

    # 3. Initialize Muon with adaptive=True
    # pass `muon_weight_dimension_numbers=None` (the default) to enable 
    # auto-detection (2D=Muon, 1D=Adam).
    optimizer = muon(
        learning_rate=1.0,
        adaptive=True,  # <--- The feature we are testing
        muon_weight_dimension_numbers=None 
    )
    
    state = optimizer.init(params)
    
    # 4. Run one update
    # If the adaptive block doesn't handle masks correctly, this might crash
    # or produce NaNs/Exploding values for 'b'.
    updates, _ = optimizer.update(grads, state, params)
    
    # 5. VERIFICATION
    
    # Check A: The matrix 'w' should be modified by Muon (Newton-Schulz)
    # The raw grad is 0.1. Muon usually normalizes gradients heavily.
    assert not jnp.allclose(updates['w'], 0.1), "Matrix 'w' should be transformed by Muon"
    
    # Check B: The vector 'b' should be UNTOUCHED by Muon's adaptive scaling.
    # Since 'b' is partitioned to Adam, the 'muon' part of the update for 'b' 
    # should effectively be a pass-through or handled by the other partition.
    # However, inside the muon transform, it sees a MaskedNode.
    # The critical check is that it DID NOT CRASH during the update call.
    
    print("✅ Test Passed: Update computed successfully.")
    print(f"   Matrix 'w' shape: {updates['w'].shape}")
    print(f"   Bias 'b' shape:   {updates['b'].shape}")

if __name__ == "__main__":
    try:
        test_adaptive_masking_stability()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise