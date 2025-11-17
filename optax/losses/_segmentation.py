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
"""Segmentation losses."""

from typing import Optional

import jax
import jax.numpy as jnp
from optax._src import utils


def dice_loss(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    class_weights: Optional[jax.typing.ArrayLike] = None,
    smooth: jax.typing.ArrayLike = 1.0,
    apply_softmax: bool = True,
    reduction: str = "mean",
    ignore_background: bool = False,
    axis: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
  r"""Computes the Dice Loss for multi-class segmentation.

  Computes the Soft Dice Loss for segmentation tasks. Works for both binary
  and multi-class segmentation. For binary segmentation, use targets with
  shape [..., 1] or [...] and predictions with corresponding logits.

  The loss is computed per class and then averaged (or summed) across classes.
  For class c:

  .. math::
    intersection_c = \sum_i^{N} p_{i,c} \cdot t_{i,c}
    \\
    dice_c = \frac{2 \cdot intersection_c + smooth}{
      \sum_i^{N} p_{i,c} + \sum_i^{N} t_{i,c} + smooth
    }

  where:
      - :math:`p_{i,c}` is the predicted probability for class c at pixel i
      - :math:`t_{i,c}` is the target value (0 or 1) for class c at pixel i
      - N is the total number of pixels

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
      axis: Axis or sequence of axes to sum over when computing the loss.
      If None, sums over all spatial dimensions (all except the first
      and last). For example, with input shape (batch, H, W, C), the
      default is to sum over H and W dimensions.

  Returns:
      Loss values. Shape depends on reduction:

      - 'mean'/'sum': [...] (batch dimensions only)
      - 'none': [..., num_classes] (includes class dimension)

  Examples:
      Binary segmentation:

      >>> import jax.numpy as jnp
      >>> from optax.losses import dice_loss
      >>> logits = jnp.array([[1.0, -1.0], [0.5, 0.5]])  # Shape: [2, 2]
      >>> targets = jnp.array([[1.0, 0.0], [1.0, 0.0]])  # Shape: [2, 2]
      >>> loss = dice_loss(logits[..., None], targets[..., None])
      >>> loss.shape
      (2,)

      Multi-class segmentation:

      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> logits = jax.random.normal(key, (2, 4, 4, 3))  # 2 samples, 3 classes
      >>> labels = jax.random.randint(key, (2, 4, 4), 0, 3)  # Random labels
      >>> targets = jax.nn.one_hot(labels, 3)  # One-hot encoded
      >>> loss = dice_loss(logits, targets)
      >>> loss.shape
      (2,)

  References:
      Milletari et al. "V-Net: Fully Convolutional Neural Networks for
      Volumetric Medical Image Segmentation" (2016).
  """

  if predictions.ndim == targets.ndim - 1:
    predictions = predictions[..., None]
  if targets.ndim == predictions.ndim - 1:
    targets = targets[..., None]
  utils.check_shapes_equal(predictions, targets)

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
  probs = predictions
  if apply_softmax:
    probs = (
        jax.nn.sigmoid(predictions)
        if predictions.shape[-1] == 1
        else jax.nn.softmax(predictions, axis=-1)
    )

  # Default behavior: sum over all spatial dimensions (except first/last)
  axis = tuple(range(1, probs.ndim - 1)) if axis is None else axis

  # Compute intersection and sums over specified axes
  intersection = jnp.sum(probs * targets, axis=axis)
  pred_sum = jnp.sum(probs, axis=axis)
  target_sum = jnp.sum(targets, axis=axis)

  # Compute Dice coefficient per class
  dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
  dice_l = 1.0 - dice_coeff  # [..., classes]

  # Apply class weights if provided
  if class_weights is not None:
    utils.check_rank(class_weights, 1)
    dice_l = dice_l * class_weights

  # Handle background class ignoring
  if ignore_background and probs.shape[-1] > 1:
    # Exclude the first class (background) from loss computation
    dice_l = dice_l[..., 1:]

  # Reduce across classes according to reduction parameter
  if reduction == "mean":
    dice_l = jnp.mean(dice_l, axis=-1)
  elif reduction == "sum":
    dice_l = jnp.sum(dice_l, axis=-1)
  elif reduction == "none":
    pass  # Keep per-class losses
  else:
    raise ValueError(
        f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
    )

  return dice_l


def multiclass_generalized_dice_loss(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    smooth: jax.typing.ArrayLike = 1.0,
    apply_softmax: bool = True,
    ignore_background: bool = False,
) -> jax.Array:
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
  utils.check_shapes_equal(predictions, targets)

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
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    smooth: jax.typing.ArrayLike = 1.0,
    apply_sigmoid: bool = True,
) -> jax.Array:
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
