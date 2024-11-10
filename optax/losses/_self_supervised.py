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
  loss = -jnp.where(matches == 1, log_softm, 0.0
                    ).sum()/matches.sum()

  return loss


def triplet_loss(
    anchors: jnp.ndarray,
    positives: jnp.ndarray,
    negatives: jnp.ndarray,
    axis: int = -1,
    p: int = 2,
    margin: float = 1.0,
    eps: float = 1e-6,
    reduction: str = 'none',
) -> jnp.ndarray:
  """
  Computes the triplet loss for a set of anchor, positive, and negative samples.
  Margin is represented with alpha in the math section.

  Args:
      anchors (array): The anchor samples.
      positives (array): The positive samples.
      negatives (array): The negative samples.
      axis (int, optional): The distribution axis. Default: -1.
      p (int, optional): The norm degree for pairwise distance. Default: 2.
      margin (float, optional): Margin for the triplet loss. Defaults to 1.0.
      eps (float, optional): Small positive constant to prevent numerical 
      instability. Defaults to 1e-6.
      reduction (str, optional): Specifies the reduction to apply to the 
      output: 'none' | 'mean' | 'sum'. Default: 'none'.

  Returns:
      array: Computed triplet loss. If reduction is 'none', returns a tensor of 
      the same shape as input;
      if reduction is 'mean' or 'sum', returns a scalar tensor.
  """
  positive_distance = jnp.sqrt(jnp.power(anchors - positives, p).sum(axis) + eps
                               )
  negative_distance = jnp.sqrt(jnp.power(anchors - negatives, p).sum(axis) + eps
                               )
  loss = jnp.maximum(positive_distance - negative_distance + margin, 0)
  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  else:
    return loss
