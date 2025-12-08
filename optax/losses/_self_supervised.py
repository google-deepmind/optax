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

import jax
from jax import lax
import jax.numpy as jnp
from optax._src import utils
from optax.losses import _regression


def ntxent(
    embeddings: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    temperature: jax.typing.ArrayLike = 0.07,
) -> jax.Array:
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
    >>> print("input:", x)  # doctest: +SKIP
    input: [[ 0.07592554 -0.48634264]
     [ 1.2903206   0.5196119 ]
     [ 0.30040437  0.31034866]
     [ 0.5761609  -0.8074621 ]]
    >>> print("labels:", labels)
    labels: [0 0 1 1]
    >>>
    >>> w = jax.random.normal(key2, shape=(2,1)) # params
    >>> b = jax.random.normal(key3, shape=(1,)) # params
    >>> out = x @ w + b # model
    >>>
    >>> print("Embeddings:", out)  # doctest: +SKIP
    Embeddings: [[0.08969027]
     [1.6291292 ]
     [0.8622629 ]
     [0.13612625]]
    >>> loss = optax.ntxent(out, labels)
    >>> print("loss:", loss)  # doctest: +SKIP
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
  utils.check_subdtype(embeddings, jnp.floating)
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


def triplet_margin_loss(
    anchors: jax.typing.ArrayLike,
    positives: jax.typing.ArrayLike,
    negatives: jax.typing.ArrayLike,
    axis: int = -1,
    norm_degree: jax.typing.ArrayLike = 2,
    margin: jax.typing.ArrayLike = 1.0,
    eps: jax.typing.ArrayLike = 1e-6,
) -> jax.Array:
  """Returns the triplet loss for a batch of embeddings.

  Examples:
    >>> import jax.numpy as jnp, optax, chex
    >>> jnp.set_printoptions(precision=4)
    >>> anchors = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    >>> positives = jnp.array([[0.1, 0.1], [1.1, 1.1]])
    >>> negatives = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> output = optax.losses.triplet_margin_loss(anchors, positives, negatives,
    ...                                           margin=1.0)
    >>> print(output)
    [0.1414 0.1414]

  Args:
    anchors: An array of anchor embeddings, with shape [batch, feature_dim].
    positives: An array of positive embeddings (similar to anchors), with
      shape [batch, feature_dim].
    negatives: An array of negative embeddings (dissimilar to anchors), with
      shape [batch, feature_dim].
    axis: The axis along which to compute the distances (default is -1).
    norm_degree: The norm degree for distance calculation (default is 2 for
      Euclidean distance).
    margin: The minimum margin by which the positive distance should be
      smaller than the negative distance.
    eps: A small epsilon value to ensure numerical stability in the distance
      calculation.

  Returns:
    Returns the computed triplet loss as an array.

  References:
      V. Balntas et al,
      `Learning shallow convolutional feature descriptors with triplet losses
      <https://bmva-archive.org.uk/bmvc/2016/papers/paper119/abstract119.pdf>`_,
      2016.
  """
  utils.check_subdtype(anchors, jnp.floating)
  utils.check_subdtype(positives, jnp.floating)
  utils.check_subdtype(negatives, jnp.floating)
  positive_distance = jnp.power(jnp.power(anchors - positives, norm_degree)
                                .sum(axis) + eps, 1/norm_degree)
  negative_distance = jnp.power(jnp.power(anchors - negatives, norm_degree)
                                .sum(axis) + eps, 1/norm_degree)
  loss = jnp.maximum(positive_distance - negative_distance + margin, 0)
  return loss
