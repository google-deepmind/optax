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
"""Segmentation losses."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp


def dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    class_weights: Optional[chex.Array] = None,
    smooth: float = 1.0,
    apply_softmax: bool = True,
    reduction: str = "mean",
    ignore_background: bool = False,
) -> chex.Array:
    """Computes the Dice Loss for multi-class segmentation.

    Computes the Soft Dice Loss for segmentation tasks. Works for both binary
    and multi-class segmentation. For binary segmentation, use targets with
    shape [..., 1] or [...] and predictions with corresponding logits.

    The loss is computed per class and then averaged (or summed) across classes.
    For class c: dice_c = (2 * intersection_c + smooth) /
                         (pred_c + target_c + smooth)

    Args:
      predictions: Logits of shape [..., num_classes] for multi-class or
                  [..., 1] or [...] for binary segmentation.
      targets: One-hot encoded targets of shape [..., num_classes] for
              multi-class or binary targets of shape [..., 1] or [...] for
              binary.
      class_weights: Optional weights for each class of shape [num_classes].
                    If None, all classes weighted equally.
      smooth: Smoothing parameter to avoid division by zero and improve
             gradient stability.
      apply_softmax: Whether to apply softmax to predictions. Set False if
                    predictions are already probabilities.
      reduction: How to reduce across classes: 'mean', 'sum', or 'none'.
                'none' returns per-class losses.
      ignore_background: If True, excludes the first class (index 0) from loss
                        computation. Useful when class 0 represents background.

    Returns:
      Loss values. Shape depends on reduction:

      - 'mean'/'sum': [...] (batch dimensions only)
      - 'none': [..., num_classes] (includes class dimension)

    Examples:
      Binary segmentation:

      >>> logits = jnp.array([[1.0, -1.0], [0.5, 0.5]])  # Shape: [2, 2]
      >>> targets = jnp.array([[1.0, 0.0], [1.0, 0.0]])  # Shape: [2, 2]
      >>> loss = dice_loss(logits[..., None], targets[..., None])

      Multi-class segmentation:

      >>> key = jax.random.PRNGKey(0)
      >>> logits = jax.random.normal(key, (4, 64, 64, 3))  # 4 samples, 3 cls
      >>> targets = jax.nn.one_hot(labels, 3)  # One-hot encoded
      >>> loss = dice_loss(logits, targets)

    References:
      Milletari et al. "V-Net: Fully Convolutional Neural Networks for
      Volumetric Medical Image Segmentation" (2016).
    """

    if predictions.ndim == targets.ndim - 1:
        predictions = predictions[..., None]
    if targets.ndim == predictions.ndim - 1:
        targets = targets[..., None]

    chex.assert_equal_shape([predictions, targets])

    # Input validation for probability distributions
    if not apply_softmax:
        # Tolerance for numerical stability
        pred_sums = jnp.sum(predictions, axis=-1)
        if not jnp.allclose(pred_sums, 1.0, rtol=1e-5):
            raise ValueError(
                "When apply_softmax=False, predictions must be valid "
                "probability distributions that sum to 1 along the class axis. "
                f"Found sum range: [{jnp.min(pred_sums):.6f}, "
                f"{jnp.max(pred_sums):.6f}]"
            )

    # Convert logits to probabilities
    if apply_softmax:
        if predictions.shape[-1] == 1:
            # Binary case - use sigmoid
            probs = jax.nn.sigmoid(predictions)
        else:
            # Multi-class case - use softmax
            probs = jax.nn.softmax(predictions, axis=-1)
    else:
        probs = predictions

    # Compute intersection and union for each class
    # Sum over all spatial dimensions, keep only batch and class dimensions
    # For input shape (batch, H, W, classes), sum over H and W dimensions
    spatial_axes = tuple(
        range(1, probs.ndim - 1)
    )  # All dimensions except batch and classes
    intersection = jnp.sum(
        probs * targets, axis=spatial_axes
    )  # [batch, classes]
    pred_sum = jnp.sum(probs, axis=spatial_axes)  # [batch, classes]
    target_sum = jnp.sum(targets, axis=spatial_axes)  # [batch, classes]

    # Compute Dice coefficient per class
    dice_coeff = (2.0 * intersection + smooth) / (
        pred_sum + target_sum + smooth
    )
    dice_loss = 1.0 - dice_coeff  # [..., classes]

    # Apply class weights if provided
    if class_weights is not None:
        num_classes = probs.shape[-1]
        chex.assert_shape(class_weights, (num_classes,))
        dice_loss = dice_loss * class_weights

    # Handle background class ignoring
    if ignore_background and probs.shape[-1] > 1:
        # Exclude the first class (background) from loss computation
        dice_loss = dice_loss[..., 1:]

    # Reduce across classes according to reduction parameter
    if reduction == "mean":
        dice_loss = jnp.mean(dice_loss, axis=-1)
    elif reduction == "sum":
        dice_loss = jnp.sum(dice_loss, axis=-1)
    elif reduction == "none":
        pass  # Keep per-class losses
    else:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        )

    return dice_loss


def multiclass_generalized_dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    smooth: float = 1.0,
    apply_softmax: bool = True,
    ignore_background: bool = False,
) -> chex.Array:
    """Computes Multiclass Generalized Dice Loss with automatic class weighting.

    Computes Generalized Dice Loss where class weights are automatically
    computed as the inverse of the squared class frequencies. This helps
    handle class imbalance in segmentation tasks.

    Args:
      predictions: Logits of shape [..., num_classes].
      targets: One-hot encoded targets of shape [..., num_classes].
      smooth: Smoothing parameter.
      apply_softmax: Whether to apply softmax to predictions.
      ignore_background: If True, excludes the first class (index 0) from loss
                        computation. Useful when class 0 represents background.

    Returns:
      Scalar loss value averaged across all classes and batch.

    References:
      Sudre et al. "Generalised Dice overlap as a deep learning loss function
      for highly unbalanced segmentations" (2017).
    """
    chex.assert_equal_shape([predictions, targets])

    # Compute class frequencies for weighting
    class_frequencies = jnp.sum(targets, axis=tuple(range(targets.ndim - 1)))

    # Compute weights as inverse of squared frequencies
    # Add small epsilon to avoid division by zero
    epsilon = 1e-7
    class_weights = 1.0 / (class_frequencies**2 + epsilon)

    # Normalize weights
    class_weights = class_weights / jnp.sum(class_weights) * len(class_weights)

    return jnp.mean(
        dice_loss(
            predictions,
            targets,
            class_weights=class_weights,
            smooth=smooth,
            apply_softmax=apply_softmax,
            reduction="none",
            ignore_background=ignore_background,
        )
    )


def binary_dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    smooth: float = 1.0,
    apply_sigmoid: bool = True,
) -> chex.Array:
    """Binary Dice Loss convenience function.

    Args:
      predictions: Logits of shape [...] or [..., 1].
      targets: Binary targets of shape [...] or [..., 1].
      smooth: Smoothing parameter.
      apply_sigmoid: Whether to apply sigmoid to predictions.

    Returns:
      Loss values of shape [...] (batch dimensions only).
    """
    # Ensure both have channel dimension
    if predictions.ndim == targets.ndim and predictions.shape[-1] != 1:
        predictions = predictions[..., None]
        targets = targets[..., None]

    return dice_loss(
        predictions,
        targets,
        smooth=smooth,
        apply_softmax=apply_sigmoid,
        reduction="mean",
    )
