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


def byol_loss(
    online_predictions: jax.typing.ArrayLike,
    target_projections: jax.typing.ArrayLike,
    axis: int = -1,
) -> jax.Array:
  """Bootstrap Your Own Latent (BYOL) loss.

  Computes the BYOL loss as the mean squared error between L2-normalized
  online predictions and target projections. This is equivalent to
  `2 - 2 * cosine_similarity`.

  Examples:
    >>> import jax
    >>> import optax
    >>> import jax.numpy as jnp
    >>>
    >>> key = jax.random.key(42)
    >>> key1, key2 = jax.random.split(key, 2)
    >>> online_pred = jax.random.normal(key1, shape=(4, 128))
    >>> target_proj = jax.random.normal(key2, shape=(4, 128))
    >>>
    >>> loss = optax.losses.byol_loss(online_pred, target_proj)
    >>> print(loss.shape)
    (4,)

  Args:
    online_predictions: Predictions from the online network with shape
      [batch, feature_dim]. These should be the output of the predictor head
      applied to the online projection.
    target_projections: Projections from the target (momentum) network with
      shape [batch, feature_dim]. Gradients should typically not flow through
      this tensor (use `jax.lax.stop_gradient`).
    axis: The axis along which to normalize and compute the loss.

  Returns:
    Per-sample BYOL loss values with shape [batch].

  References:
    J.-B. Grill et al, `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_, 2020.

  .. versionadded:: 0.2.4
  """
  utils.check_subdtype(online_predictions, jnp.floating)
  utils.check_subdtype(target_projections, jnp.floating)

  # L2 normalize both predictions and projections
  eps = jnp.finfo(online_predictions.dtype).eps
  online_norm = online_predictions / (
      jnp.linalg.norm(online_predictions, axis=axis, keepdims=True) + eps
  )
  target_norm = target_projections / (
      jnp.linalg.norm(target_projections, axis=axis, keepdims=True) + eps
  )

  # MSE between normalized vectors = 2 - 2 * cosine_similarity
  loss = 2.0 - 2.0 * jnp.sum(online_norm * target_norm, axis=axis)
  return loss


def simsiam_loss(
    predictions: jax.typing.ArrayLike,
    projections: jax.typing.ArrayLike,
    axis: int = -1,
) -> jax.Array:
  """Simple Siamese (SimSiam) loss.

  Computes the negative cosine similarity between predictions and projections.
  Note: The stop-gradient operation should be applied to `projections` before
  calling this function.

  Examples:
    >>> import jax
    >>> import optax
    >>> import jax.numpy as jnp
    >>>
    >>> key = jax.random.key(42)
    >>> key1, key2 = jax.random.split(key, 2)
    >>> p1 = jax.random.normal(key1, shape=(4, 128))  # predictions
    >>> z2 = jax.random.normal(key2, shape=(4, 128))  # projections
    >>>
    >>> # Standard usage with stop_gradient on projections
    >>> loss = optax.losses.simsiam_loss(p1, jax.lax.stop_gradient(z2))
    >>> print(loss.shape)
    (4,)

  Args:
    predictions: Output of the predictor network with shape [batch,
      feature_dim].
    projections: Output of the projection network for the other view with shape
      [batch, feature_dim]. Should have stop_gradient applied externally.
    axis: The axis along which to compute the cosine similarity.

  Returns:
    Per-sample SimSiam loss values with shape [batch].

  References:
    X. Chen et al, `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_, 2021.

  .. versionadded:: 0.2.4
  """
  utils.check_subdtype(predictions, jnp.floating)
  utils.check_subdtype(projections, jnp.floating)

  # L2 normalize both vectors
  eps = jnp.finfo(predictions.dtype).eps
  p_norm = predictions / (
      jnp.linalg.norm(predictions, axis=axis, keepdims=True) + eps
  )
  z_norm = projections / (
      jnp.linalg.norm(projections, axis=axis, keepdims=True) + eps
  )

  # Negative cosine similarity
  loss = -jnp.sum(p_norm * z_norm, axis=axis)
  return loss


def dino_loss(
    student_logits: jax.typing.ArrayLike,
    teacher_logits: jax.typing.ArrayLike,
    teacher_temp: jax.typing.ArrayLike = 0.04,
    student_temp: jax.typing.ArrayLike = 0.1,
    center: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Self-DIstillation with NO labels (DINO) loss.

  Computes the cross-entropy between sharpened teacher and student outputs.
  The teacher logits can be centered to avoid collapse.

  Examples:
    >>> import jax
    >>> import optax
    >>> import jax.numpy as jnp
    >>>
    >>> key = jax.random.key(42)
    >>> key1, key2 = jax.random.split(key, 2)
    >>> student = jax.random.normal(key1, shape=(4, 256))
    >>> teacher = jax.random.normal(key2, shape=(4, 256))
    >>>
    >>> # Without centering
    >>> loss = optax.losses.dino_loss(student, teacher)
    >>> print(loss.shape)
    (4,)
    >>>
    >>> # With centering (center should be EMA of teacher outputs)
    >>> center = jnp.zeros(256)
    >>> loss = optax.losses.dino_loss(student, teacher, center=center)
    >>> print(loss.shape)
    (4,)

  Args:
    student_logits: Logits from the student network with shape [batch, dim].
    teacher_logits: Logits from the teacher network with shape [batch, dim].
      Gradients should typically not flow through this tensor.
    teacher_temp: Temperature for sharpening the teacher distribution. Lower
      values produce sharper distributions.
    student_temp: Temperature for the student distribution.
    center: Optional center vector to subtract from teacher logits to prevent
      collapse. Shape [dim]. Should be an exponential moving average of teacher
      outputs.

  Returns:
    Per-sample DINO loss values with shape [batch].

  References:
    M. Caron et al, `Emerging Properties in Self-Supervised Vision Transformers
    <https://arxiv.org/abs/2104.14294>`_, 2021.

  .. versionadded:: 0.2.4
  """
  utils.check_subdtype(student_logits, jnp.floating)
  utils.check_subdtype(teacher_logits, jnp.floating)

  # Apply centering to teacher logits if provided
  if center is not None:
    teacher_logits = teacher_logits - center

  # Compute softmax probabilities with temperature
  teacher_probs = jax.nn.softmax(teacher_logits / teacher_temp, axis=-1)
  student_log_probs = jax.nn.log_softmax(student_logits / student_temp, axis=-1)

  # Cross-entropy loss: -sum(teacher_probs * log(student_probs))
  loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
  return loss


def barlow_twins_loss(
    embeddings_a: jax.typing.ArrayLike,
    embeddings_b: jax.typing.ArrayLike,
    lambda_: jax.typing.ArrayLike = 5e-3,
) -> jax.Array:
  """Barlow Twins loss for self-supervised learning.

  Computes the Barlow Twins loss which encourages the cross-correlation matrix
  between two sets of embeddings to be close to the identity matrix. This
  promotes invariance to augmentations (diagonal elements close to 1) while
  reducing redundancy (off-diagonal elements close to 0).

  Examples:
    >>> import jax
    >>> import optax
    >>> import jax.numpy as jnp
    >>>
    >>> key = jax.random.key(42)
    >>> key1, key2 = jax.random.split(key, 2)
    >>> z_a = jax.random.normal(key1, shape=(32, 128))  # batch of embeddings
    >>> z_b = jax.random.normal(key2, shape=(32, 128))  # from augmented views
    >>>
    >>> loss = optax.losses.barlow_twins_loss(z_a, z_b)
    >>> print(loss.shape)
    ()

  Args:
    embeddings_a: Embeddings from the first augmented view with shape [batch,
      feature_dim].
    embeddings_b: Embeddings from the second augmented view with shape [batch,
      feature_dim].
    lambda_: Trade-off parameter for the redundancy reduction term. Higher
      values put more weight on decorrelating different dimensions.

  Returns:
    A scalar loss value.

  References:
    J. Zbontar et al, `Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction <https://arxiv.org/abs/2103.06573>`_, 2021.

  .. versionadded:: 0.2.4
  """
  utils.check_subdtype(embeddings_a, jnp.floating)
  utils.check_subdtype(embeddings_b, jnp.floating)

  # Normalize embeddings along the batch dimension (zero mean, unit std)
  eps = jnp.finfo(embeddings_a.dtype).eps
  z_a = (embeddings_a - jnp.mean(embeddings_a, axis=0)) / (
      jnp.std(embeddings_a, axis=0) + eps
  )
  z_b = (embeddings_b - jnp.mean(embeddings_b, axis=0)) / (
      jnp.std(embeddings_b, axis=0) + eps
  )

  batch_size = embeddings_a.shape[0]

  # Compute cross-correlation matrix
  # Shape: [feature_dim, feature_dim]
  c = jnp.einsum('bi,bj->ij', z_a, z_b) / batch_size

  # Invariance term: (1 - C_ii)^2 for diagonal elements
  on_diag = jnp.sum(jnp.square(1.0 - jnp.diag(c)))

  # Redundancy reduction term: C_ij^2 for off-diagonal elements
  # Create mask for off-diagonal elements
  dim = c.shape[0]
  off_diag_mask = 1.0 - jnp.eye(dim)
  off_diag = jnp.sum(jnp.square(c) * off_diag_mask)

  loss = on_diag + lambda_ * off_diag
  return loss
