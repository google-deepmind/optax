"""Segmentation losses."""

from typing import Optional

import chex
import jax
# import jax.numpy as jnp

from optax.losses import sigmoid_binary_cross_entropy

def sigmoid_focal_loss(
    logits:  chex.Array,
    labels:  chex.Array,
    alpha: Optional[float] = None,
    gamma: float = 2,
) ->  chex.Array:
  """Compute a sigmoid focal loss as proposed by Lin et al.
  This loss often appears in the segmentation context.
  Use this loss function if classes are not mutually exclusive.
  See `sigmoid_binary_cross_entropy` for more information.
  References:
    Lin et al. https://arxiv.org/pdf/1708.02002.pdf
  
  Args:
    logits: A float array of arbitrary shape.
      The predictions for each example.
    labels: A float array, its shape must be identical to
      that of logits. It containes the binary
      classification label for each element in logits
      (0 for the out of class and 1 for in class).
      This array is often one-hot encoded.
    alpha: (optional) Weighting factor in range (0,1) to balance
      positive vs negative examples. Default None (no weighting).
    gamma: Exponent of the modulating factor (1 - p_t) to
      balance easy vs hard examples.

  Returns:
    A loss value array with a shape identical to the logits and target
    arrays.
  """
  chex.assert_type([logits], float)
  labels = labels.astype(logits.dtype)
  # see also the original papers implementation at:
  # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
  p = jax.nn.sigmoid(logits)
  ce_loss = sigmoid_binary_cross_entropy(logits, labels)
  p_t = p * labels + (1 - p) * (1 - labels)
  loss = ce_loss * ((1 - p_t) ** gamma)
  if alpha:
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss = alpha_t * loss
  return loss
