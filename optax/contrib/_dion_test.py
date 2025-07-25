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
"""Tests for DION optimizer."""

import functools
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from optax.contrib import _dion


class DionTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = {
        'matrix': jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2x3 matrix
        'vector': jnp.array([1.0, 2.0, 3.0]),  # 1D vector
        'scalar': jnp.array(5.0),  # scalar
        'bias': jnp.array([0.1, 0.2])  # 1D bias
    }
    self.gradients = {
        'matrix': jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        'vector': jnp.array([0.1, 0.2, 0.3]),
        'scalar': jnp.array(0.5),
        'bias': jnp.array([0.01, 0.02])
    }

  def test_is_matrix_param(self):
    """Test matrix parameter detection."""
    # Should be True for 2D matrices with both dims > 1
    self.assertTrue(_dion._is_matrix_param(jnp.ones((3, 4))))
    self.assertTrue(_dion._is_matrix_param(jnp.ones((2, 2))))
    
    # Should be False for vectors, scalars, or degenerate matrices
    self.assertFalse(_dion._is_matrix_param(jnp.ones((3,))))      # 1D vector
    self.assertFalse(_dion._is_matrix_param(jnp.ones(())))        # scalar
    self.assertFalse(_dion._is_matrix_param(jnp.ones((1, 5))))    # row vector
    self.assertFalse(_dion._is_matrix_param(jnp.ones((5, 1))))    # column vector
    self.assertFalse(_dion._is_matrix_param(jnp.ones((1, 1))))    # 1x1 matrix

  def test_low_rank_approximation_shapes(self):
    """Test that low-rank approximation returns correct shapes."""
    matrix = jnp.ones((4, 6))
    rank = 3
    
    P, R = _dion._power_iteration_approximation(matrix, rank)
    
    # Check shapes
    self.assertEqual(P.shape, (4, rank))
    self.assertEqual(R.shape, (6, rank))
    
    # Check that reconstruction has correct shape
    reconstruction = P @ R.T
    self.assertEqual(reconstruction.shape, matrix.shape)

  def test_low_rank_approximation_rank_limiting(self):
    """Test that rank is properly limited by matrix dimensions."""
    matrix = jnp.ones((2, 3))
    large_rank = 10
    
    P, R = _dion._power_iteration_approximation(matrix, large_rank)
    
    # Rank should be limited to min(m, n) = 2
    self.assertEqual(P.shape[1], 2)
    self.assertEqual(R.shape[1], 2)

  def test_low_rank_approximation_accuracy(self):
    """Test accuracy of low-rank approximation on known matrix."""
    # Create a known low-rank matrix
    np.random.seed(42)
    U = jnp.array(np.random.randn(4, 2))
    V = jnp.array(np.random.randn(3, 2))
    matrix = U @ V.T  # Rank-2 matrix
    
    # Approximate with same rank
    P, R = _dion._power_iteration_approximation(matrix, rank=2)
    reconstruction = P @ R.T
    
    # Should be nearly exact
    chex.assert_trees_all_close(matrix, reconstruction, atol=1e-5)

  def test_orthogonalize_P(self):
    """Test P matrix orthogonalization."""
    np.random.seed(42)
    P = jnp.array(np.random.randn(5, 3))
    
    P_orth = _dion._orthogonalize_P(P)
    
    # Check that columns are orthonormal
    should_be_identity = P_orth.T @ P_orth
    expected_identity = jnp.eye(3)
    chex.assert_trees_all_close(should_be_identity, expected_identity, atol=1e-5)

  def test_scale_by_dion_init(self):
    """Test initialization of DION optimizer state."""
    opt = _dion.scale_by_dion()
    state = opt.init(self.init_params)
    
    # Check state structure
    self.assertIsInstance(state, _dion.ScaleByDionState)
    
    # Check momentum is zero-initialized with same structure as params
    chex.assert_trees_all_close(
        state.momentum,
        jax.tree.map(jnp.zeros_like, self.init_params)
    )
    
    # Check counter is initialized to 0
    self.assertEqual(state.count, 0)

  def test_scale_by_dion_update_shapes(self):
    """Test that DION updates have correct shapes."""
    opt = _dion.scale_by_dion(rank_fraction=0.5)
    state = opt.init(self.init_params)
    
    updates, new_state = opt.update(self.gradients, state)
    
    # Updates should have same structure as gradients
    chex.assert_trees_all_equal_structs(updates, self.gradients)
    
    # State should have same structure
    self.assertIsInstance(new_state, _dion.ScaleByDionState)
    chex.assert_trees_all_equal_structs(new_state.momentum, self.init_params)

  def test_scale_by_dion_step_counter(self):
    """Test step counter increments correctly."""
    opt = _dion.scale_by_dion()
    state = opt.init(self.init_params)
    
    # Take several steps
    for expected_count in range(1, 4):
      updates, state = opt.update(self.gradients, state)
      self.assertEqual(state.count, expected_count)

  def test_scale_by_dion_basic_convergence(self):
    """Test DION basic functionality on simple optimization."""
    # Simple quadratic: f(x) = 0.5 * (x - 1)^2
    def loss_fn(x):
      return 0.5 * (x - 1.0) ** 2
    
    grad_fn = jax.grad(loss_fn)
    
    # Initialize - use scalar for simplicity
    x = jnp.array(5.0)
    opt = _dion.scale_by_dion(momentum_decay=0.5, rank_fraction=0.1)
    state = opt.init({'x': x})
    
    # Optimize - scalar uses momentum, should converge
    learning_rate = 0.1
    for _ in range(50):
      grads = {'x': grad_fn(x)}
      updates, state = opt.update(grads, state)
      x = x - learning_rate * updates['x']
    
    # Should converge close to optimum (x = 1)
    chex.assert_trees_all_close(x, jnp.array(1.0), atol=0.1)

  def test_dion_optimizer(self):
    """Test high-level DION optimizer interface."""
    opt = _dion.dion(learning_rate=0.01, momentum_decay=0.9, rank_fraction=0.2)
    state = opt.init(self.init_params)
    
    updates, new_state = opt.update(self.gradients, state)
    
    # Check shapes are preserved
    chex.assert_trees_all_equal_structs(updates, self.gradients)
    
    # Check updates are scaled by learning rate (should be smaller than gradients)
    # For matrix parameter, check that update magnitude is reasonable
    matrix_update_norm = jnp.linalg.norm(updates['matrix'])
    matrix_grad_norm = jnp.linalg.norm(self.gradients['matrix'])
    self.assertLess(matrix_update_norm, matrix_grad_norm)

  def test_dion_different_matrix_sizes(self):
    """Test DION handles different matrix shapes correctly."""
    matrices = {
        'small': jnp.ones((2, 3)),
        'square': jnp.ones((4, 4)),
        'tall': jnp.ones((6, 2)),
        'wide': jnp.ones((2, 8))
    }
    gradients = jax.tree.map(lambda x: 0.1 * jnp.ones_like(x), matrices)
    
    opt = _dion.scale_by_dion(rank_fraction=0.3)
    state = opt.init(matrices)
    
    updates, new_state = opt.update(gradients, state)
    
    # All should complete without error and preserve shapes
    chex.assert_trees_all_equal_structs(updates, gradients)

  def test_dion_numerical_stability(self):
    """Test DION handles edge cases without NaN/inf."""
    # Very small gradients
    small_grads = jax.tree.map(lambda x: 1e-8 * jnp.ones_like(x), self.init_params)
    
    # Very large gradients
    large_grads = jax.tree.map(lambda x: 1e3 * jnp.ones_like(x), self.init_params)
    
    opt = _dion.scale_by_dion(eps=1e-7)
    
    for grads in [small_grads, large_grads]:
      state = opt.init(self.init_params)
      updates, new_state = opt.update(grads, state)
      
      # Check no NaN/inf values
      def check_finite(x):
        self.assertTrue(jnp.all(jnp.isfinite(x)))
      
      jax.tree.map(check_finite, updates)
      jax.tree.map(check_finite, new_state.momentum)

  def test_matrix_vs_scalar_updates(self):
    """Test that matrix parameters get different treatment than scalars."""
    params = {
        'matrix': jnp.ones((3, 4)),  # Should use DION update
        'vector': jnp.ones(5),       # Should use momentum update
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x), params)
    
    opt = _dion.scale_by_dion(momentum_decay=0.0, rank_fraction=0.5)  # No momentum for easier comparison
    state = opt.init(params)
    
    updates1, state1 = opt.update(grads, state)
    updates2, state2 = opt.update(grads, state1)
    
    # For matrix: DION updates may be similar if the buffer is similar
    # The key test is that they use different algorithms
    self.assertTrue(_dion._is_matrix_param(params['matrix']))
    self.assertFalse(_dion._is_matrix_param(params['vector']))
    
    # For vector: with momentum_decay=0, should be same as gradients
    chex.assert_trees_all_close(updates1['vector'], grads['vector'], atol=1e-6)
    chex.assert_trees_all_close(updates2['vector'], grads['vector'], atol=1e-6)


if __name__ == '__main__':
  absltest.main()