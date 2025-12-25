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
"""Tests for the _mask_callable integration in Muon optimizer.

These tests demonstrate that the TODO fix correctly integrates
_masking._mask_callable for determining when weight_dimension_numbers
should be called as a function.
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from optax.contrib import _muon
from optax.transforms import _masking


class MaskCallableIntegrationTest(absltest.TestCase):
  """Tests to verify the _mask_callable integration in scale_by_shape and scale_by_muon."""

  def test_scale_by_shape_with_callable_weight_dim_nums(self):
    """Test that scale_by_shape correctly handles a callable weight_dimension_numbers.
    
    This is the key test for the TODO fix - demonstrating that _mask_callable
    properly identifies when weight_dimension_numbers is a callable.
    """
    # Define a callable that returns dimension numbers based on parameter structure
    def weight_dim_nums_fn(params):
      return jax.tree.map(
          lambda x: _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None,
          params
      )

    # Create the transformation with callable weight_dimension_numbers
    transform = _muon.scale_by_shape(
        weight_dimension_numbers=weight_dim_nums_fn,
        consistent_rms=None,
    )

    # Test with mixed parameters (2D and 1D)
    params = {
        'w1': jnp.ones((10, 10)),  # 2D - should be scaled
        'b1': jnp.ones((10,)),      # 1D - should NOT be scaled (None -> identity)
    }

    state = transform.init(params)
    updates = params  # Use params as updates for simplicity
    scaled_updates, _ = transform.update(updates, state, params=params)

    # Verify that 2D parameter is scaled (scale = sqrt(max(1, fan_out/fan_in)) = 1 for 10x10)
    # For square matrices with width scaling, the scale factor is 1
    self.assertEqual(scaled_updates['w1'].shape, updates['w1'].shape)
    # Verify that 1D parameter is NOT scaled (identity for None)
    self.assertTrue(jnp.allclose(scaled_updates['b1'], updates['b1']))

  def test_scale_by_shape_with_non_square_matrices(self):
    """Test scaling with non-square matrices where scale factor differs from 1."""
    def weight_dim_nums_fn(params):
      return jax.tree.map(
          lambda x: _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None,
          params
      )

    transform = _muon.scale_by_shape(
        weight_dimension_numbers=weight_dim_nums_fn,
        consistent_rms=None,
    )

    # Non-square matrix: fan_in=5, fan_out=20, scale = sqrt(max(1, 20/5)) = 2
    params = {
        'w1': jnp.ones((5, 20)),   # 2D - scale = sqrt(4) = 2
        'b1': jnp.ones((20,)),      # 1D - identity
    }

    state = transform.init(params)
    updates = params
    scaled_updates, _ = transform.update(updates, state, params=params)

    # w1 should be scaled by sqrt(20/5) = 2
    expected_scale = jnp.sqrt(jnp.maximum(1, 20 / 5))  # = 2.0
    self.assertTrue(jnp.allclose(scaled_updates['w1'], updates['w1'] * expected_scale))
    # b1 should be identity
    self.assertTrue(jnp.allclose(scaled_updates['b1'], updates['b1']))

  def test_scale_by_shape_with_static_weight_dim_nums(self):
    """Test that scale_by_shape correctly handles static (non-callable) weight_dimension_numbers."""
    # Static dimension numbers
    dim_nums = {
        'w1': _muon.MuonDimensionNumbers(0, 1),
        'w2': _muon.MuonDimensionNumbers(0, 1),
    }

    transform = _muon.scale_by_shape(
        weight_dimension_numbers=dim_nums,
        consistent_rms=None,
    )

    params = {
        'w1': jnp.ones((5, 20)),   # scale = 2
        'w2': jnp.ones((10, 10)),  # scale = 1
    }

    state = transform.init(params)
    updates = params
    scaled_updates, _ = transform.update(updates, state, params=params)

    # w1 should be scaled by 2
    expected_scale_w1 = jnp.sqrt(jnp.maximum(1, 20 / 5))  # = 2.0
    self.assertTrue(jnp.allclose(scaled_updates['w1'], updates['w1'] * expected_scale_w1))
    # w2 should be scaled by 1
    self.assertTrue(jnp.allclose(scaled_updates['w2'], updates['w2']))

  def test_scale_by_shape_handles_masked_node(self):
    """Test that scale_by_shape correctly handles MaskedNode leaves."""
    # Define a callable that returns MaskedNode for some parameters
    def weight_dim_nums_fn(params):
      return {
          'w1': _muon.MuonDimensionNumbers(0, 1),  # Normal scaling
          'w2': _masking.MaskedNode(),  # Should be treated as identity
      }

    transform = _muon.scale_by_shape(
        weight_dimension_numbers=weight_dim_nums_fn,
    )

    params = {
        'w1': jnp.ones((5, 20)),  # scale = 2
        'w2': jnp.ones((5, 5)),   # identity (MaskedNode)
    }

    state = transform.init(params)
    updates = params
    scaled_updates, _ = transform.update(updates, state, params=params)

    # w1 should be scaled by 2
    expected_scale = jnp.sqrt(jnp.maximum(1, 20 / 5))  # = 2.0
    self.assertTrue(jnp.allclose(scaled_updates['w1'], updates['w1'] * expected_scale))
    # w2 should NOT be scaled (MaskedNode -> identity)
    self.assertTrue(jnp.allclose(scaled_updates['w2'], updates['w2']))

  def test_muon_with_callable_weight_dim_nums_and_masking(self):
    """Test full muon optimizer with callable weight_dimension_numbers and masking."""
    def weight_dim_nums_fn(params):
      # Return dimension numbers only for 2D params, None for others
      return jax.tree.map(
          lambda x: _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None,
          params
      )

    opt = _muon.muon(
        learning_rate=1e-3,
        muon_weight_dimension_numbers=weight_dim_nums_fn,
    )

    # Mixed parameter types
    params = {
        'linear': {
            'w': jnp.ones((8, 8)),   # 2D - muon applies
            'b': jnp.ones((8,)),      # 1D - adam applies
        },
        'embed': jnp.ones((10, 16)),  # 2D - muon applies
    }

    state = opt.init(params)
    updates = params  # Use params as gradients for test
    new_updates, new_state = opt.update(updates, state, params=params)

    # All updates should be modified (muon or adam)
    self.assertFalse(jnp.allclose(new_updates['linear']['w'], updates['linear']['w']))
    self.assertFalse(jnp.allclose(new_updates['linear']['b'], updates['linear']['b']))
    self.assertFalse(jnp.allclose(new_updates['embed'], updates['embed']))

  def test_mask_callable_correctly_identifies_callables(self):
    """Test that _mask_callable correctly identifies various inputs.
    
    This demonstrates the core utility function that the TODO fix relies on.
    """
    # A function is callable
    self.assertTrue(_masking._mask_callable(lambda x: x))

    # A static pytree is NOT callable (leaves are not all callable)
    static_tree = {'a': 1, 'b': 2}
    self.assertFalse(_masking._mask_callable(static_tree))

    # MuonDimensionNumbers is NOT callable
    dim_nums = _muon.MuonDimensionNumbers(0, 1)
    self.assertFalse(_masking._mask_callable(dim_nums))

    # A tree of MuonDimensionNumbers is NOT callable
    dim_nums_tree = {'w': _muon.MuonDimensionNumbers(0, 1)}
    self.assertFalse(_masking._mask_callable(dim_nums_tree))
  def test_scale_by_shape_root_none_implies_2d_matrices(self):
    """Test that root-level None implies all parameters are 2D matrices.
    
    As per the docstring: 'None implies that all parameters are 2D matrices.'
    This means root-level None should use default MuonDimensionNumbers(0, 1).
    """
    # Root-level None - should apply default dimension numbers
    transform = _muon.scale_by_shape(
        weight_dimension_numbers=None,
        consistent_rms=None,
    )

    # Non-square 2D matrix: fan_in=5, fan_out=20, scale = sqrt(4) = 2
    params = {'w': jnp.ones((5, 20))}

    state = transform.init(params)
    updates = params
    scaled_updates, _ = transform.update(updates, state, params=params)

    # Should be scaled with default MuonDimensionNumbers(0, 1)
    expected_scale = jnp.sqrt(jnp.maximum(1, 20 / 5))  # = 2.0
    self.assertTrue(jnp.allclose(scaled_updates['w'], updates['w'] * expected_scale))

  def test_scale_by_shape_leaf_none_skips_parameter(self):
    """Test that leaf-level None in pytree means skip that parameter (identity).
    
    This is different from root-level None. A None inside the dim_nums pytree
    means 'do not apply scaling to this parameter'.
    """
    # Leaf-level None in pytree - w2 should be skipped
    dim_nums = {
        'w1': _muon.MuonDimensionNumbers(0, 1),  # Apply scaling
        'w2': None,  # Skip scaling (identity)
    }

    transform = _muon.scale_by_shape(
        weight_dimension_numbers=dim_nums,
        consistent_rms=None,
    )

    params = {
        'w1': jnp.ones((5, 20)),  # scale = 2
        'w2': jnp.ones((5, 20)),  # identity (leaf None)
    }

    state = transform.init(params)
    updates = params
    scaled_updates, _ = transform.update(updates, state, params=params)

    # w1 should be scaled
    expected_scale = jnp.sqrt(jnp.maximum(1, 20 / 5))  # = 2.0
    self.assertTrue(jnp.allclose(scaled_updates['w1'], updates['w1'] * expected_scale))
    # w2 should NOT be scaled (leaf-level None -> identity)
    self.assertTrue(jnp.allclose(scaled_updates['w2'], updates['w2']))


if __name__ == '__main__':
  absltest.main()
