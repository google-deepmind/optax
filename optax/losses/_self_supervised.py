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

  Examples:
    >>> import jax
    >>> import optax
    >>> import jax.numpy as jnp
    >>>
    >>> key = jax.random.key(42)
    >>> key1, key2, key3 = jax.random.split(key, 3)
    >>> x = jax.random.normal(key1, shape=(4,2))
    >>> labels = jnp.array([0, 0, 1, 1])
    >>>
    >>> print("input:", x)
    input: [[-0.9155995   1.5534698 ]
     [ 0.2623586  -1.5908985 ]
     [-0.15977189  0.480501  ]
     [ 0.58389133  0.10497775]]
    >>> print("labels:", labels)
    labels: [0 0 1 1]
    >>>
    >>> w = jax.random.normal(key2, shape=(2,1)) # params
    >>> b = jax.random.normal(key3, shape=(1,)) # params
    >>> out = x @ w + b # model
    >>>
    >>> print("Embeddings:", out)
    Embeddings: [[-1.0076267]
     [-1.2960069]
     [-1.1829865]
     [-1.3485558]]
    >>> loss = optax.ntxent(out, labels)
    >>> print("loss:", loss)
    loss: 1.0986123

  Args:
    embeddings: batch of embeddings, with shape [batch, feature_length]
    labels: labels for groups that are positive pairs. e.g. if you have a batch
      of 4 embeddings and the first two and last two were positive pairs your
      `labels` should look like [0, 0, 1, 1]. Shape [batch]
    temperature: temperature scaling parameter.

  Returns:
    A scalar loss value of NT-Xent values averaged over all positive
    pairs

  References:
    T. Chen et al `A Simple Framework for Contrastive Learning of Visual
    Representations <http://arxiv.org/abs/2002.05709>`_, 2020

    kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss

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
          embeddings[None, :, :], embeddings[:, None, :],
          epsilon=jnp.finfo(embeddings.dtype).eps
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
  loss = -jnp.where(matches == 1, log_softm, 0.0).sum()/matches.sum()

  return loss


def triplet_loss(
    anchors: chex.Array,
    positives: chex.Array,
    negatives: chex.Array,
    axis: int = -1,
    norm_degree: chex.Numeric = 2,
    margin: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-6,
) -> chex.Array:
  """Computes the triplet loss for a batch of embeddings.

    Examples:
      >>> import jax.numpy as jnp
      >>> import optax
      >>> import chex
      >>> anchors = jnp.array([[0.0, 0.0], [1.0, 1.0]])
      >>> positives = jnp.array([[0.1, 0.1], [1.1, 1.1]])
      >>> negatives = jnp.array([[1.0, 0.0], [0.0, 1.0]])
      >>> output =optax.triplet_loss(anchors, positives, negatives, margin=1.0)
      >>> print(output)
      >>> Array([0.14142442, 0.14142442], dtype=float32)

    Args:
        anchors: An array of anchor embeddings, with shape [batch, feature_dim].
        positives: An array of positive embeddings
          (similar to anchors), with shape [batch, feature_dim].
        negatives: An array of negative embeddings
          (dissimilar to anchors), with shape [batch, feature_dim].
        axis: The axis along which to compute the distances
          (default is -1).
        p: The norm degree for distance calculation
          (default is 2 for Euclidean distance).
        margin: The minimum margin by which the positive distance
          should be smaller than the negative distance.
        eps: A small epsilon value to ensure numerical stability
          in the distance calculation.
        reduction: Specifies the reduction to apply to the
          output: 'none' | 'mean' | 'sum'.

    Returns:
        The computed triplet loss as an array or scalar
        depending on the reduction parameter. 
        If reduction is 'mean' or 'sum', returns a scalar.

    References:
        V. Balntas et al, 
        `Learning shallow convolutional feature descriptors with triplet losses 
        <https://bmva-archive.org.uk/bmvc/2016/papers/paper119/abstract119.pdf>`
        _, 2016
    """
  chex.assert_type([anchors, positives, negatives], float)
  positive_distance = jnp.power(jnp.power(anchors - positives, norm_degree)
                                .sum(axis) + eps, 1/norm_degree)
  negative_distance = jnp.power(jnp.power(anchors - negatives, norm_degree)
                                .sum(axis) + eps, 1/norm_degree)
  loss = jnp.maximum(positive_distance - negative_distance + margin, 0)
  return loss
