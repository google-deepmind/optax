# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Self supervised losses."""

import chex
from jax import lax
import jax.numpy as jnp
from optax.losses import _regression


def ntxent(
    embeddings: chex.Array, labels: chex.Array, temperature: chex.Numeric = 0.07
) -> chex.Numeric:
  """Normalized temperature scaled cross entropy loss (NT-Xent).

  References:
    T. Chen et al `A Simple Framework for Contrastive Learning of Visual
    Representations <http://arxiv.org/abs/2002.05709>`_, 2020
    kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss

  Args:
    embeddings: batch of embeddings, with shape [batch, feature_length]
    labels: labels for groups that are positive pairs. e.g. if you have a batch
      of 4 embeddings and the first two and last two were positive pairs your
      `labels` should look like [0, 0, 1, 1]. labels SHOULD NOT be all the same
      (e.g. [0, 0, 0, 0]) you will get a NaN result. Shape [batch]
    temperature: temperature scaling parameter.

  Returns:
    A scalar loss value of NT-Xent values averaged over all positive
    pairs

  .. versionadded:: 0.2.3
  """
  chex.assert_type([embeddings], float)
  if labels.shape[0] != embeddings.shape[0]:
    raise ValueError(
        'Labels and embeddings must have the same leading dimension, found'
        f' {labels.shape[0]} for labels and {embeddings.shape[0]} for'
        ' embeddings.'
    )

  # cosine similarity matrix
  xcs = (
      _regression.cosine_similarity(
          embeddings[None, :, :], embeddings[:, None, :]
      )
      / temperature
  )

  # finding positive and negative pairs
  labels1 = jnp.expand_dims(labels, axis=1)
  labels2 = jnp.expand_dims(labels, axis=0)
  matches = labels1 == labels2
  diffs = matches ^ 1
  matches = jnp.bool_(matches - jnp.eye(matches.shape[0]))  # no self cos

  # replace 0 with -inf
  xcs_diffs = jnp.where(diffs == 1, xcs, -jnp.inf)
  xcs_matches = jnp.where(matches == 1, xcs, -jnp.inf)

  # shifting for numeric stability
  comb = jnp.concatenate((xcs_diffs, xcs_matches), axis=-1)
  xcs_max = jnp.max(comb, axis=1, keepdims=True)
  xcs_shift_diffs = xcs_diffs - lax.stop_gradient(xcs_max)
  xcs_shift_matches = xcs_matches - lax.stop_gradient(xcs_max)

  # calc loss
  numer = xcs_shift_matches
  numer_exp = jnp.exp(xcs_shift_matches)
  denom = jnp.sum(jnp.exp(xcs_shift_diffs), axis=1, keepdims=True)
  denom += numer_exp
  log_softm = numer - jnp.log(denom)
  loss = -jnp.where(matches == 1, log_softm, 0.0).sum() / matches.sum()

  return loss


def _pairwise_distance(x: chex.Array, y: chex.Array, p: int = 2, eps: float = 1e-6) -> chex.Array:
    diff = x - y
    dist = jnp.sum(jnp.abs(diff) ** p + eps, axis=-1) ** (1.0 / p)
    return dist
  

def triplet_margin_loss(
    anchor: chex.Array,
    positive: chex.Array,
    negative: chex.Array,
    *,
    margin: float = 1.0,
    p: int = 2,
    eps: float = 1e-6,
    swap: bool = False,
    reduction: str = 'mean',
) -> chex.Array:
    """Triplet margin loss function.
    
    Measures the relative similarity between an anchor point, a positive point, and
    a negative point using the distance metric specified by p-norm. The loss encourages
    the distance between the anchor and positive points to be smaller than the distance
    between the anchor and negative points by at least the margin amount.
    
    Args:
        anchor: The anchor embeddings. Shape: [batch_size, feature_dim].
        positive: The positive embeddings. Shape: [batch_size, feature_dim].
        negative: The negative embeddings. Shape: [batch_size, feature_dim].
        margin: The margin value. Default: 1.0.
        p: The norm degree for pairwise distance. Default: 2.
        eps: Small epsilon value to avoid numerical issues. Default: 1e-6.
        swap: Use the distance swap optimization from "Learning shallow convolutional 
            feature descriptors with triplet losses" by V. Balntas et al. Default: False.
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    
    Returns:
        The triplet margin loss value.
        If reduction is 'none': tensor of shape [batch_size]
        If reduction is 'mean' or 'sum': scalar tensor.
    """
    chex.assert_equal_shape([anchor, positive, negative])

    if not(anchor.ndim ==2 and positive.ndim ==2 and negative.ndim ==2):
        raise ValueError("Inputs must be 2D tensors")
    
    # Calculate distances between pairs
    dist_pos = _pairwise_distance(anchor, positive, p, eps)
    dist_neg = _pairwise_distance(anchor, negative, p, eps)
    
    # Implement distance swap if enabled
    if swap:
        dist_swap = _pairwise_distance(positive, negative)
        dist_neg = jnp.minimum(dist_neg, dist_swap)
    
    # Calculate loss with margin
    losses = jnp.maximum(margin + dist_pos - dist_neg, 0.0)
    
    # Apply reduction
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return jnp.mean(losses)
    elif reduction == 'sum':
        return jnp.sum(losses)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
    