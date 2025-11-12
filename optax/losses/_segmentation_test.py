# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for segmentation losses in `optax.losses._segmentation.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses import _segmentation


class DiceLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(42)

  def test_dice_loss_shapes(self):
    """Test that the loss function handles various input shapes."""
    key = self.key

    # Binary segmentation shapes
    binary_cases = [
        ((4, 64, 64, 1), (4, 64, 64, 1)),  # Standard binary
        ((2, 32, 32, 32, 1), (2, 32, 32, 32, 1)),  # 3D binary
        ((1, 128, 128, 1), (1, 128, 128, 1)),  # Single sample
    ]

    for pred_shape, target_shape in binary_cases:
      with self.subTest(pred_shape=pred_shape, target_shape=target_shape):
        predictions = jax.random.normal(key, pred_shape)
        targets = jax.random.bernoulli(key, 0.5, target_shape)

        loss = _segmentation.dice_loss(predictions, targets)
        # Expected shape: batch dimensions only (default
        # reduction='mean')
        # For shape (batch, spatial..., classes), output is (batch,)
        # We need to keep only the first dimension (batch)
        expected_shape = pred_shape[:1]  # Keep only batch dimension
        self.assertEqual(loss.shape, expected_shape)
        self.assertTrue(jnp.isfinite(loss).all())

    # Multi-class segmentation shapes
    multiclass_cases = [
        ((4, 64, 64, 3), (4, 64, 64, 3)),  # 3 classes
        ((2, 32, 32, 5), (2, 32, 32, 5)),  # 5 classes
        ((1, 128, 128, 10), (1, 128, 128, 10)),  # 10 classes
    ]

    for pred_shape, target_shape in multiclass_cases:
      with self.subTest(pred_shape=pred_shape, target_shape=target_shape):
        predictions = jax.random.normal(key, pred_shape)
        # Create proper one-hot targets
        class_labels = jax.random.randint(
            key, target_shape[:-1], 0, target_shape[-1]
        )
        targets = jax.nn.one_hot(class_labels, target_shape[-1])

        loss = _segmentation.dice_loss(predictions, targets)
        # Expected shape: batch dimensions only (default
        # reduction='mean')
        # For shape (batch, spatial..., classes), output is (batch,)
        # We need to keep only the first dimension (batch)
        expected_shape = pred_shape[:1]  # Keep only batch dimension
        self.assertEqual(loss.shape, expected_shape)
        self.assertTrue(jnp.isfinite(loss).all())

  def test_perfect_overlap_binary(self):
    """Test that perfect predictions give loss close to 0."""
    # Create perfect binary predictions
    targets = jnp.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])[
        ..., None
    ]  # Shape: (1, 3, 3, 1)

    # Perfect logits (high confidence)
    perfect_logits = jnp.where(targets, 10.0, -10.0)

    loss = _segmentation.dice_loss(perfect_logits, targets, smooth=1.0)

    # Loss should be very close to 0 but not exactly due to smoothing
    self.assertLess(jnp.mean(loss), 0.01)

  def test_no_overlap_binary(self):
    """Test that completely wrong predictions give high loss."""
    targets = jnp.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])[
        ..., None
    ]  # Shape: (1, 3, 3, 1)

    # Completely wrong logits
    wrong_logits = jnp.where(targets, -10.0, 10.0)

    loss = _segmentation.dice_loss(wrong_logits, targets, smooth=1.0)

    # Loss should be high (but may not reach 0.9 due to smoothing)
    # With smooth=1.0, even perfect mismatch won't reach 1.0
    self.assertGreater(jnp.mean(loss), 0.7)

  def test_perfect_overlap_multiclass(self):
    """Test perfect predictions for multi-class case."""
    # Create one-hot targets
    class_labels = jnp.array([[0, 1, 2], [1, 0, 2], [2, 2, 1]])  # Shape: (3, 3)
    targets = jax.nn.one_hot(class_labels, 3)[None, ...]  # Shape: (1, 3, 3, 3)

    # Perfect logits
    perfect_logits = jnp.where(targets, 10.0, -10.0)

    loss = _segmentation.dice_loss(perfect_logits, targets, smooth=1.0)

    # Loss should be very close to 0
    self.assertLess(jnp.mean(loss), 0.01)

  def test_loss_bounds(self):
    """Test that loss is always between 0 and 1."""
    key = self.key

    for _ in range(10):  # Test multiple random cases
      key, subkey = jax.random.split(key)

      # Random binary case
      predictions = jax.random.normal(subkey, (2, 16, 16, 1))
      targets = jax.random.bernoulli(subkey, 0.3, (2, 16, 16, 1))

      loss = _segmentation.dice_loss(predictions, targets)

      self.assertTrue(jnp.all(loss >= 0.0))
      self.assertTrue(jnp.all(loss <= 1.0))

      # Random multi-class case
      predictions = jax.random.normal(subkey, (2, 16, 16, 4))
      class_labels = jax.random.randint(subkey, (2, 16, 16), 0, 4)
      targets = jax.nn.one_hot(class_labels, 4)

      loss = _segmentation.dice_loss(predictions, targets)

      self.assertTrue(jnp.all(loss >= 0.0))
      self.assertTrue(jnp.all(loss <= 1.0))

  @parameterized.parameters(
      {"smooth": 0.0},
      {"smooth": 1.0},
      {"smooth": 10.0},
  )
  def test_smoothing_parameter(self, smooth):
    """Test different smoothing parameter values."""
    predictions = jax.random.normal(self.key, (2, 8, 8, 1))
    targets = jax.random.bernoulli(self.key, 0.5, (2, 8, 8, 1))

    loss = _segmentation.dice_loss(predictions, targets, smooth=smooth)

    self.assertTrue(jnp.isfinite(loss).all())
    self.assertTrue(jnp.all(loss >= 0.0))
    self.assertTrue(jnp.all(loss <= 1.0))

  def test_class_weights(self):
    """Test class weighting functionality."""
    predictions = jax.random.normal(self.key, (2, 16, 16, 3))
    class_labels = jax.random.randint(self.key, (2, 16, 16), 0, 3)
    targets = jax.nn.one_hot(class_labels, 3)

    # Test with equal weights (should be same as no weights)
    equal_weights = jnp.ones(3)
    loss_equal = _segmentation.dice_loss(
        predictions, targets, class_weights=equal_weights
    )
    loss_none = _segmentation.dice_loss(
        predictions, targets, class_weights=None
    )

    np.testing.assert_allclose(loss_equal, loss_none, rtol=1e-6)

    # Test with different weights
    custom_weights = jnp.array([1.0, 2.0, 0.5])
    loss_weighted = _segmentation.dice_loss(
        predictions, targets, class_weights=custom_weights
    )

    # Should be different from unweighted
    self.assertFalse(jnp.allclose(loss_weighted, loss_none))

  @parameterized.parameters(
      {"reduction": "mean"},
      {"reduction": "sum"},
      {"reduction": "none"},
  )
  def test_reduction_modes(self, reduction):
    """Test different reduction modes."""
    predictions = jax.random.normal(self.key, (2, 16, 16, 3))
    class_labels = jax.random.randint(self.key, (2, 16, 16), 0, 3)
    targets = jax.nn.one_hot(class_labels, 3)

    loss = _segmentation.dice_loss(predictions, targets, reduction=reduction)

    if reduction == "none":
      # Should keep class dimension: batch_dims + [num_classes]
      # For (2, 16, 16, 3), sum over spatial dims gives (2, 3)
      self.assertEqual(loss.shape, (2, 3))
    else:
      # Should reduce class dimension: just batch_dims
      # For (2, 16, 16, 3), sum over spatial dims and reduce classes
      # gives (2,)
      self.assertEqual(loss.shape, (2,))

    self.assertTrue(jnp.isfinite(loss).all())

  def test_gradient_computation(self):
    """Test that gradients can be computed successfully."""
    predictions = jax.random.normal(self.key, (1, 8, 8, 1))
    targets = jax.random.bernoulli(self.key, 0.3, (1, 8, 8, 1))

    def loss_fn(preds):
      return jnp.mean(_segmentation.dice_loss(preds, targets))

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(predictions)

    # Gradients should have same shape as predictions
    self.assertEqual(grads.shape, predictions.shape)
    # Gradients should be finite
    self.assertTrue(jnp.isfinite(grads).all())
    # Gradients shouldn't all be zero (unless in very special cases)
    self.assertTrue(jnp.any(grads != 0.0))

  def test_numerical_stability(self):
    """Test numerical stability with extreme values."""
    # Very large logits
    large_predictions = jnp.full((1, 4, 4, 1), 100.0)
    targets = jnp.ones((1, 4, 4, 1))

    loss_large = _segmentation.dice_loss(large_predictions, targets)
    self.assertTrue(jnp.isfinite(loss_large).all())

    # Very small logits
    small_predictions = jnp.full((1, 4, 4, 1), -100.0)

    loss_small = _segmentation.dice_loss(small_predictions, targets)
    self.assertTrue(jnp.isfinite(loss_small).all())

  def test_apply_softmax_flag(self):
    """Test apply_softmax flag functionality."""
    logits = jax.random.normal(self.key, (1, 8, 8, 3))
    class_labels = jax.random.randint(self.key, (1, 8, 8), 0, 3)
    targets = jax.nn.one_hot(class_labels, 3)

    # With softmax (default)
    loss_with_softmax = _segmentation.dice_loss(
        logits, targets, apply_softmax=True
    )

    # Without softmax (pass probabilities directly)
    probs = jax.nn.softmax(logits, axis=-1)
    loss_without_softmax = _segmentation.dice_loss(
        probs, targets, apply_softmax=False
    )

    # Should be approximately equal
    np.testing.assert_allclose(
        loss_with_softmax, loss_without_softmax, rtol=1e-6
    )

  def test_binary_dice_loss_convenience(self):
    """Test the binary dice loss convenience function."""
    # Test with shape [...] (no channel dimension)
    predictions_2d = jax.random.normal(self.key, (2, 16, 16))
    targets_2d = jax.random.bernoulli(self.key, 0.3, (2, 16, 16))

    loss_2d = _segmentation.binary_dice_loss(predictions_2d, targets_2d)
    # For shape (2, 16, 16) -> (2, 16, 16, 1), sum over spatial dims
    # gives (2,)
    self.assertEqual(loss_2d.shape, (2,))
    self.assertTrue(jnp.isfinite(loss_2d).all())

    # Test with shape [..., 1] (with channel dimension)
    predictions_3d = predictions_2d[..., None]
    targets_3d = targets_2d[..., None]

    loss_3d = _segmentation.binary_dice_loss(predictions_3d, targets_3d)

    # Should give similar results
    np.testing.assert_allclose(loss_2d, loss_3d, rtol=1e-5)

  def test_multiclass_generalized_dice_loss(self):
    """Test the generalized dice loss with automatic weighting."""
    predictions = jax.random.normal(self.key, (4, 32, 32, 3))

    # Create imbalanced targets (class 0 dominant)
    class_probs = jnp.array([0.8, 0.15, 0.05])  # Imbalanced classes
    class_labels = jax.random.choice(self.key, 3, (4, 32, 32), p=class_probs)
    targets = jax.nn.one_hot(class_labels, 3)

    gdl_loss = _segmentation.multiclass_generalized_dice_loss(
        predictions, targets
    )
    regular_loss = jnp.mean(_segmentation.dice_loss(predictions, targets))

    # Should be a scalar
    self.assertEqual(gdl_loss.shape, ())
    self.assertTrue(jnp.isfinite(gdl_loss))

    # Should be different from regular dice loss due to weighting
    self.assertFalse(jnp.allclose(gdl_loss, regular_loss))

  def test_edge_cases(self):
    """Test edge cases."""
    # All zeros target
    predictions = jax.random.normal(self.key, (1, 4, 4, 1))
    targets = jnp.zeros((1, 4, 4, 1))

    loss = _segmentation.dice_loss(predictions, targets, smooth=1.0)
    self.assertTrue(jnp.isfinite(loss).all())

    # All ones target
    targets = jnp.ones((1, 4, 4, 1))
    loss = _segmentation.dice_loss(predictions, targets, smooth=1.0)
    self.assertTrue(jnp.isfinite(loss).all())

  def test_shape_mismatch_error(self):
    """Test that shape mismatch raises appropriate errors."""
    predictions = jax.random.normal(self.key, (2, 16, 16, 3))
    targets = jax.random.bernoulli(
        self.key, 0.5, (2, 16, 16, 2)
    )  # Wrong classes

    with self.assertRaises(ValueError):
      _segmentation.dice_loss(predictions, targets)

  def test_jit_compilation(self):
    """Test that the function can be JIT compiled."""
    # Use simpler shapes to avoid dynamic reshape issues
    predictions = jax.random.normal(
        self.key, (2, 8, 1)
    )  # Shape: [batch, spatial, classes]
    targets = jax.random.bernoulli(self.key, 0.3, (2, 8, 1))

    # Test without JIT first
    loss_no_jit = _segmentation.dice_loss(predictions, targets)

    # Test with JIT
    jit_loss_fn = jax.jit(_segmentation.dice_loss)
    loss_jit = jit_loss_fn(predictions, targets)

    # Should be approximately equal
    np.testing.assert_allclose(loss_no_jit, loss_jit, rtol=1e-6)

  def test_vmap_compatibility(self):
    """Test that the function works with vmap."""

    def single_loss(pred, target):
      return _segmentation.dice_loss(pred[None, ...], target[None, ...])

    batched_loss = jax.jit(jax.vmap(single_loss))

    predictions = jax.random.normal(
        self.key, (4, 4, 1)
    )  # Simpler shape: [batch, spatial, classes]
    targets = jax.random.bernoulli(self.key, 0.3, (4, 4, 1))

    # Should work with vmap
    losses = batched_loss(predictions, targets)
    # For input (1, 4, 1), sum over spatial dims gives (1, 1), reduce
    # classes gives (1,)
    # So vmapped over 4 samples gives (4, 1)
    self.assertEqual(losses.shape, (4, 1))
    self.assertTrue(jnp.isfinite(losses).all())

  def test_ignore_background_parameter(self):
    """Test ignore_background parameter functionality."""
    # Create multi-class predictions and targets
    predictions = jax.random.normal(self.key, (2, 8, 8, 3))
    class_labels = jax.random.randint(self.key, (2, 8, 8), 0, 3)
    targets = jax.nn.one_hot(class_labels, 3)

    # Test with background included (default)
    loss_with_bg = _segmentation.dice_loss(
        predictions, targets, ignore_background=False, reduction="none"
    )
    # Should have shape (2, 3) - all classes
    self.assertEqual(loss_with_bg.shape, (2, 3))

    # Test with background ignored
    loss_no_bg = _segmentation.dice_loss(
        predictions, targets, ignore_background=True, reduction="none"
    )
    # Should have shape (2, 2) - only foreground classes
    self.assertEqual(loss_no_bg.shape, (2, 2))

    # Check that foreground classes have the same loss values
    # The loss values for classes 1 and 2 should be the same whether
    # background is ignored or not
    self.assertTrue(jnp.allclose(loss_with_bg[..., 1:], loss_no_bg))

    # Test with binary segmentation - ignore_background should have no
    # effect
    binary_predictions = jax.random.normal(self.key, (2, 8, 8, 1))
    binary_targets = jax.random.bernoulli(self.key, 0.5, (2, 8, 8, 1))

    loss_binary_with_bg = _segmentation.dice_loss(
        binary_predictions, binary_targets, ignore_background=False
    )
    loss_binary_no_bg = _segmentation.dice_loss(
        binary_predictions, binary_targets, ignore_background=True
    )
    # Should be the same for binary case
    self.assertTrue(jnp.allclose(loss_binary_with_bg, loss_binary_no_bg))

  def test_improved_shape_handling(self):
    """Test the improved shape handling for binary cases."""
    key = self.key

    # Test case 1: predictions have no channel dim, targets have channel dim
    predictions_no_ch = jax.random.normal(key, (2, 8, 8))  # No channel
    targets_with_ch = jax.random.bernoulli(
        key, 0.5, (2, 8, 8, 1)
    )  # With channel

    loss1 = _segmentation.dice_loss(predictions_no_ch, targets_with_ch)
    self.assertEqual(loss1.shape, (2,))
    self.assertTrue(jnp.isfinite(loss1).all())

    # Test case 2: predictions have channel dim, targets have no channel dim
    predictions_with_ch = jax.random.normal(key, (2, 8, 8, 1))  # With channel
    targets_no_ch = jax.random.bernoulli(key, 0.5, (2, 8, 8))  # No channel

    loss2 = _segmentation.dice_loss(predictions_with_ch, targets_no_ch)
    self.assertEqual(loss2.shape, (2,))
    self.assertTrue(jnp.isfinite(loss2).all())

    # Test case 3: both have matching shapes
    predictions_match = jax.random.normal(key, (2, 8, 8, 1))
    targets_match = jax.random.bernoulli(key, 0.5, (2, 8, 8, 1))

    loss3 = _segmentation.dice_loss(predictions_match, targets_match)
    self.assertEqual(loss3.shape, (2,))
    self.assertTrue(jnp.isfinite(loss3).all())

  def test_probability_validation(self):
    """Test validation when apply_softmax=False."""
    key = self.key

    # Test with valid probabilities
    logits = jax.random.normal(key, (2, 8, 8, 3))
    valid_probs = jax.nn.softmax(logits, axis=-1)
    targets = jax.random.bernoulli(key, 0.3, (2, 8, 8, 3))

    # Should work fine
    loss = _segmentation.dice_loss(valid_probs, targets, apply_softmax=False)
    self.assertTrue(jnp.isfinite(loss).all())

    # Test with invalid probabilities (don't sum to 1)
    invalid_probs = jax.random.uniform(key, (2, 8, 8, 3))  # Random values

    # Should raise ValueError
    with self.assertRaises(ValueError) as context:
      _segmentation.dice_loss(invalid_probs, targets, apply_softmax=False)

    self.assertIn("valid probability distributions", str(context.exception))

  def test_generalized_dice_loss_ignore_background(self):
    """Test ignore_background parameter in generalized dice loss."""
    predictions = jax.random.normal(self.key, (2, 16, 16, 4))
    class_labels = jax.random.randint(self.key, (2, 16, 16), 0, 4)
    targets = jax.nn.one_hot(class_labels, 4)

    # Test with and without background
    gdl_with_bg = _segmentation.multiclass_generalized_dice_loss(
        predictions, targets, ignore_background=False
    )
    gdl_no_bg = _segmentation.multiclass_generalized_dice_loss(
        predictions, targets, ignore_background=True
    )

    # Should be different values
    self.assertFalse(jnp.allclose(gdl_with_bg, gdl_no_bg))
    self.assertTrue(jnp.isfinite(gdl_with_bg))
    self.assertTrue(jnp.isfinite(gdl_no_bg))


if __name__ == "__main__":
  absltest.main()
