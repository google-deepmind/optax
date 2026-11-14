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
import jax.numpy as jnp
from optax._src import utils


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
  embeddings = jnp.asarray(embeddings)
  labels = jnp.asarray(labels)
  if labels.shape[0] != embeddings.shape[0]:
    raise ValueError(
        'Labels and embeddings must have the same leading dimension, found'
        f' {labels.shape[0]} for labels and {embeddings.shape[0]} for'
        ' embeddings.'
    )

  out_dtype = jnp.result_type(embeddings, temperature)
  acc_dtype = jnp.float64 if out_dtype == jnp.float64 else jnp.float32
  n = labels.shape[0]

  # cosine similarity matrix
  emb_acc = embeddings.astype(acc_dtype)
  eps = jnp.asarray(jnp.finfo(acc_dtype).eps, dtype=acc_dtype)
  norm2 = jnp.sum(jnp.square(emb_acc), axis=-1, keepdims=True)
  u = emb_acc * jax.lax.rsqrt(jnp.maximum(norm2, eps))

  inv_temp = jnp.asarray(1.0, dtype=acc_dtype) / jnp.asarray(
      temperature, dtype=acc_dtype
  )
  matmul_precision = (
      None
      if embeddings.dtype in (jnp.bfloat16, jnp.float16)
      else jax.lax.Precision.HIGHEST
  )
  xcs = (
      jnp.matmul(
          u,
          jnp.swapaxes(u, -1, -2),
          precision=matmul_precision,
          preferred_element_type=acc_dtype,
      )
      * inv_temp
  )

  # finding positive and negative pairs
  equal = labels[:, None] == labels[None, :]
  diffs = ~equal
  matches = equal & ~jnp.eye(n, dtype=bool)  # no self cos

  xcs_diffs = jnp.where(diffs, xcs, -jnp.inf)
  m = jax.lax.stop_gradient(jnp.max(xcs_diffs, axis=-1, keepdims=True))
  has_diffs = m > -jnp.inf
  m_safe = jax.lax.stop_gradient(jnp.where(has_diffs, m, 0.0))

  sum_diffs = jnp.sum(jnp.exp(xcs_diffs - m_safe), axis=-1, keepdims=True)
  log_sum_diffs = jnp.log(jnp.where(has_diffs, sum_diffs, 1.0))

  active = matches & has_diffs
  logits = jnp.where(active, (m_safe - xcs) + log_sum_diffs, 0.0)
  pair_loss = jnp.where(active, jax.nn.softplus(logits), 0.0)
  loss = jnp.sum(pair_loss) / jnp.sum(matches)
  return loss.astype(out_dtype)


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
    >>> import jax.numpy as jnp, optax
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
  positive_distance = jnp.power(
      jnp.power(
          # pyrefly: ignore[unsupported-operation]
          anchors - positives,
          norm_degree,
      ).sum(axis)
      + eps,
      1 / norm_degree,
  )
  negative_distance = jnp.power(
      jnp.power(
          # pyrefly: ignore[unsupported-operation]
          anchors - negatives,
          norm_degree,
      ).sum(axis)
      + eps,
      1 / norm_degree,
  )
  loss = jnp.maximum(positive_distance - negative_distance + margin, 0)
  return loss
