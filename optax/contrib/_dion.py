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
"""DION: Distributed Orthonormalized Updates optimizer.

Implementation of the DION optimizer from "Dion: Distributed Orthonormalized
Updates" by Ahn et al. (https://arxiv.org/abs/2504.05295).

DION provides orthonormal matrix updates with low-rank approximation for
communication efficiency in distributed training. Key features:
- Orthonormalized updates for matrix parameters
- Low-rank approximation for reduced communication overhead
- Error feedback momentum to maintain accuracy
- Element-wise updates for scalar parameters (biases, norms, etc.)
"""

from typing import NamedTuple, Optional, Union
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform


class ScaleByDionState(NamedTuple):
  """State for the DION optimizer transformation.
  
  Attributes:
    momentum: Momentum buffer (error feedback from low-rank approximation)
    count: Step counter for bias correction (if needed)
    orthonormal_basis: Previous orthonormal basis for warm-start (matrices only)
  """
  momentum: base.Updates
  count: chex.Array  # []
  orthonormal_basis: base.Updates


def _is_matrix_param(param: chex.Array) -> bool:
  """Check if parameter should be treated as a matrix (2D with both dims > 1)."""
  return param.ndim == 2 and param.shape[0] > 1 and param.shape[1] > 1


def _power_iteration_approximation(
    matrix: chex.Array,
    rank: int,
    prev_Q: Optional[chex.Array] = None,
    num_iterations: int = 1,
    eps: float = 1e-7
) -> tuple[chex.Array, chex.Array]:
  """Compute low-rank approximation using power iteration with warm-start.
  
  This follows the DION paper's approach using single power iteration
  with warm-start from previous orthonormal basis.
  
  Args:
    matrix: Input matrix to approximate [m, n]
    rank: Target rank for approximation
    prev_Q: Previous orthonormal basis [m, r] for warm-start
    num_iterations: Number of power iterations (paper uses 1)
    eps: Small constant for numerical stability
    
  Returns:
    P: Orthonormal left factor [m, r] 
    R: Right factor [n, r]
    
  The approximation satisfies: matrix ≈ P @ R.T
  """
  m, n = matrix.shape
  effective_rank = min(rank, min(m, n))
  
  # Initialize Q (left basis) with warm-start if available
  if prev_Q is not None and prev_Q.shape == (m, effective_rank):
    Q = prev_Q
  else:
    # Cold start: random initialization
    key = jax.random.PRNGKey(0)
    Q = jax.random.normal(key, (m, effective_rank))
    Q, _ = jnp.linalg.qr(Q)
  
  # Power iteration (typically just 1 iteration as per paper)
  for _ in range(num_iterations):
    # Forward pass: R = Q.T @ matrix
    R = Q.T @ matrix  # [r, n]
    R = R.T  # [n, r] to match expected output format
    
    # Backward pass: Z = matrix @ R
    Z = matrix @ R  # [m, r]
    
    # Orthogonalize via QR decomposition
    Q, _ = jnp.linalg.qr(Z)
  
  # Final computation of R factor
  R = Q.T @ matrix  # [r, n]
  R = R.T  # [n, r]
  
  return Q, R


def _orthogonalize_P(P: chex.Array, eps: float = 1e-7) -> chex.Array:
  """Orthogonalize the P matrix using QR decomposition.
  
  Args:
    P: Input matrix [m, r]
    eps: Small constant for numerical stability
    
  Returns:
    Orthogonalized P matrix [m, r]
  """
  Q, R = jnp.linalg.qr(P)
  
  # Handle potential sign ambiguity in QR
  signs = jnp.sign(jnp.diag(R))
  Q = Q * signs[None, :]
  
  return Q


def scale_by_dion(
    momentum_decay: float = 0.95,
    rank_fraction: float = 0.25,
    eps: float = 1e-7,
) -> base.GradientTransformation:
  """Scale gradients using DION optimizer.
  
  DION (Distributed Orthonormalized Updates) provides orthonormal matrix 
  updates with low-rank approximation for communication efficiency.
  
  The algorithm maintains a momentum buffer and applies:
  1. For matrix parameters: orthonormalized updates via low-rank approximation
  2. For scalar parameters: element-wise momentum updates (like SGD momentum)
  
  Algorithm:
    B_t = M_{t-1} + G_t                    # Buffer formation
    P_t, R_t = low_rank_approx(B_t, r)     # Low-rank approximation  
    M_t = B_t - (1-μ) * P_t @ R_t.T       # Error feedback momentum
    P_t = orthogonalize(P_t)               # Orthogonalization
    updates = P_t @ Q_t.T                  # Final updates (Q_t from R_t)
  
  Args:
    momentum_decay: Momentum decay rate μ ∈ [0,1). Higher values retain more
      momentum. Typically 0.9-0.99.
    rank_fraction: Fraction of matrix dimensions to use for low-rank 
      approximation. Controls communication vs accuracy tradeoff.
    eps: Small constant for numerical stability.
      
  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    momentum = jax.tree.map(jnp.zeros_like, params)
    
    # Initialize orthonormal basis for matrix parameters
    def init_basis(param):
      if _is_matrix_param(param):
        m, n = param.shape
        target_rank = max(1, int(rank_fraction * min(m, n)))
        return jnp.zeros((m, target_rank))
      else:
        return jnp.zeros_like(param)
    
    orthonormal_basis = jax.tree.map(init_basis, params)
    
    return ScaleByDionState(
        momentum=momentum, 
        count=jnp.zeros([], dtype=jnp.int32),
        orthonormal_basis=orthonormal_basis
    )

  def update_fn(updates, state, params=None):
    del params  # Unused
    
    count_inc = numerics.safe_int32_increment(state.count)
    
    def update_matrix(grad, momentum_prev, prev_basis):
      """Update rule for matrix parameters using DION."""
      m, n = grad.shape
      target_rank = max(1, int(rank_fraction * min(m, n)))
      
      # B_t = M_{t-1} + G_t (buffer formation)
      buffer = momentum_prev + grad
      
      # Low-rank approximation using power iteration with warm-start
      prev_Q = prev_basis if prev_basis.shape == (m, target_rank) else None
      P, R = _power_iteration_approximation(buffer, target_rank, prev_Q, eps=eps)
      
      # Error feedback momentum: M_t = B_t - (1-μ) * P_t @ R_t.T
      reconstruction = P @ R.T
      momentum_new = buffer - (1 - momentum_decay) * reconstruction
      
      # Final update: P is already orthogonal from power iteration
      # Apply scaling factor as per paper: √(m/n)
      scale_factor = jnp.sqrt(m / n) if n > 0 else 1.0
      update = scale_factor * P @ R.T
      
      return update, momentum_new, P
      
    def update_scalar(grad, momentum_prev, prev_basis):
      """Update rule for scalar/vector parameters using momentum."""
      momentum_new = momentum_decay * momentum_prev + grad
      return momentum_new, momentum_new, prev_basis  # Keep basis unchanged
    
    # Apply appropriate update rule based on parameter shape
    def compute_update(grad, momentum_prev, prev_basis):
      if _is_matrix_param(grad):
        update, _, _ = update_matrix(grad, momentum_prev, prev_basis)
        return update
      else:
        update, _, _ = update_scalar(grad, momentum_prev, prev_basis)
        return update
        
    def compute_momentum(grad, momentum_prev, prev_basis):
      if _is_matrix_param(grad):
        _, momentum, _ = update_matrix(grad, momentum_prev, prev_basis)
        return momentum
      else:
        _, momentum, _ = update_scalar(grad, momentum_prev, prev_basis)
        return momentum
        
    def compute_basis(grad, momentum_prev, prev_basis):
      if _is_matrix_param(grad):
        _, _, basis = update_matrix(grad, momentum_prev, prev_basis)
        return basis
      else:
        _, _, basis = update_scalar(grad, momentum_prev, prev_basis)
        return basis
    
    # Apply updates separately
    updates_new = jax.tree.map(compute_update, updates, state.momentum, state.orthonormal_basis)
    momentum_new = jax.tree.map(compute_momentum, updates, state.momentum, state.orthonormal_basis)
    basis_new = jax.tree.map(compute_basis, updates, state.momentum, state.orthonormal_basis)
    
    return updates_new, ScaleByDionState(
        momentum=momentum_new, 
        count=count_inc,
        orthonormal_basis=basis_new
    )

  return base.GradientTransformation(init_fn, update_fn)


def dion(
    learning_rate: base.ScalarOrSchedule,
    momentum_decay: float = 0.95,
    rank_fraction: float = 0.25,
    eps: float = 1e-7,
) -> base.GradientTransformation:
  """The DION optimizer.
  
  DION (Distributed Orthonormalized Updates) is designed for efficient 
  distributed training of large neural networks. It provides orthonormal
  matrix updates with low-rank approximation to reduce communication overhead.
  
  Key benefits:
  - Orthonormalized updates improve training stability
  - Low-rank approximation reduces communication in distributed training
  - Error feedback maintains accuracy despite approximation
  - Supports mixed matrix/scalar parameter handling
  
  Args:
    learning_rate: Learning rate, either fixed or scheduled.
    momentum_decay: Momentum decay rate μ ∈ [0,1). Controls momentum retention.
    rank_fraction: Fraction of min(m,n) to use for low-rank approximation rank.
      Lower values reduce communication but may hurt accuracy.
    eps: Small constant for numerical stability.
    
  Returns:
    A `GradientTransformation` object.
    
  References:
    Ahn et al., "Dion: Distributed Orthonormalized Updates" 
    https://arxiv.org/abs/2504.05295
  """
  return combine.chain(
      scale_by_dion(
          momentum_decay=momentum_decay,
          rank_fraction=rank_fraction,
          eps=eps
      ),
      transform.scale(-learning_rate)
  )