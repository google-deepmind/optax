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
  loss = -jnp.where(matches == 1, log_softm, 0.0).sum() / matches.sum()

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
  positive_distance = jnp.power(jnp.power(anchors - positives, norm_degree)
                                .sum(axis) + eps, 1 / norm_degree)
  negative_distance = jnp.power(jnp.power(anchors - negatives, norm_degree)
                                .sum(axis) + eps, 1 / norm_degree)
  loss = jnp.maximum(positive_distance - negative_distance + margin, 0)
  return loss

def byol_loss(
    online_projection_1: jax.typing.ArrayLike,
    target_projection_2: jax.typing.ArrayLike,
    online_projection_2: jax.typing.ArrayLike | None = None,
    target_projection_1: jax.typing.ArrayLike | None = None,
    eps: jax.typing.ArrayLike = 1e-6,
    symmetric: bool | None = None,
) -> jax.Array:
  """Bootstrap Your Own Latent (BYOL) loss.

  Computes the BYOL regression loss between an online network prediction and a
  target (teacher) network projection using two augmented views of each sample.

  For L2-normalized vectors q and z, the regression loss satisfies:

    ||q - z||^2 = 2 - 2 * cosine_similarity(q, z)

  Modes
  -----
  1. Single-direction:
     Uses only (q1, z2) and returns:
       mean(2 - 2 * cos(q1, stop_grad(z2)))

  2. Symmetric two-view:
     Uses both directions (q1 vs z2) and (q2 vs z1) and returns:
       mean(0.5 * [(2 - 2 * cos(q1, stop_grad(z2)))
                 + (2 - 2 * cos(q2, stop_grad(z1)))])

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> q1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> z2 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> out = optax.losses.byol_loss(q1, z2)

    >>> q2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    >>> z1 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    >>> out = optax.losses.byol_loss(q1, z2, q2, z1, symmetric=True)

  Args:
    online_projection_1: Online-network prediction for view 1 (q1),
      shape [batch, feature_dim].
    target_projection_2: Target-network projection for view 2 (z2),
      shape [batch, feature_dim].
    online_projection_2: Optional online-network prediction for view 2 (q2),
      shape [batch, feature_dim]. Required when symmetric mode is enabled.
    target_projection_1: Optional target-network projection for view 1 (z1),
      shape [batch, feature_dim]. Required when symmetric mode is enabled.
    eps: Small epsilon used inside cosine similarity for numerical stability.
    symmetric: If True, computes the symmetric BYOL loss using both directions
      (q1 vs z2 and q2 vs z1). If False, or if left as None and the second pair
      is not provided, computes only the single-direction loss between
      `online_projection_1` and `target_projection_2`. If left as None and a
      second pair (`online_projection_2` / `target_projection_1`) is provided,
      symmetric mode is enabled automatically.

  Returns:
    A scalar BYOL loss averaged over the batch. In symmetric mode, the loss is
    averaged over both directions and the batch.

  References:
    [1] J.-B. Grill et al,
        `Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
        <https://arxiv.org/abs/2006.07733>`_, 2020.
  """


  online_projection_1 = jnp.asarray(online_projection_1)
  target_projection_2 = jnp.asarray(target_projection_2)

  utils.check_subdtype(online_projection_1, jnp.floating)
  utils.check_subdtype(target_projection_2, jnp.floating)

  # Cast eps to match the projection dtype for numerical consistency.
  eps = jnp.asarray(eps, dtype=online_projection_1.dtype)

  # Decide symmetric mode in a jit-safe way.
  # If `symmetric` is a plain Python bool, respect it.
  # Otherwise (None or JAX tracer), infer from presence of second view.
  if symmetric is None or not isinstance(symmetric, bool):
    symmetric_flag = (
        online_projection_2 is not None or target_projection_1 is not None
    )
  else:
    symmetric_flag = symmetric

  # Single-direction mode: only (q1, z2) is used.
  if not symmetric_flag:
    if online_projection_1.shape != target_projection_2.shape:
      raise ValueError(
          'online_projection_1 and target_projection_2 must have the same '
          f'shape, found {online_projection_1.shape} and '
          f'{target_projection_2.shape}.'
      )

    # Stop gradient on target branch (teacher network).
    target_projection_2 = lax.stop_gradient(target_projection_2)

    cos_12 = _regression.cosine_similarity(
        online_projection_1,
        target_projection_2,
        epsilon=eps,
    )
    loss_12 = 2.0 - 2.0 * cos_12
    return jnp.mean(loss_12)

  # Symmetric mode: requires all four projections.
  if online_projection_2 is None or target_projection_1 is None:
    raise ValueError(
        'Symmetric BYOL loss requested, but `online_projection_2` or '
        '`target_projection_1` is None. Provide all four projections or '
        'set `symmetric=False`.'
    )

  online_projection_2 = jnp.asarray(online_projection_2)
  target_projection_1 = jnp.asarray(target_projection_1)

  utils.check_subdtype(online_projection_2, jnp.floating)
  utils.check_subdtype(target_projection_1, jnp.floating)

  if online_projection_1.shape != target_projection_2.shape:
    raise ValueError(
        'online_projection_1 and target_projection_2 must have the same '
        f'shape, found {online_projection_1.shape} and '
        f'{target_projection_2.shape}.'
    )
  if online_projection_2.shape != target_projection_1.shape:
    raise ValueError(
        'online_projection_2 and target_projection_1 must have the same '
        f'shape, found {online_projection_2.shape} and '
        f'{target_projection_1.shape}.'
    )

  # Stop gradient on target branch (teacher network).
  target_projection_1 = lax.stop_gradient(target_projection_1)
  target_projection_2 = lax.stop_gradient(target_projection_2)

  # BYOL uses squared L2 distance between L2-normalized vectors.
  # For normalized vectors: ||q - z||^2 = 2 - 2 * cos(q, z).
  cos_12 = _regression.cosine_similarity(
      online_projection_1,
      target_projection_2,
      epsilon=eps,
  )
  cos_21 = _regression.cosine_similarity(
      online_projection_2,
      target_projection_1,
      epsilon=eps,
  )

  loss_12 = 2.0 - 2.0 * cos_12
  loss_21 = 2.0 - 2.0 * cos_21
  loss = 0.5 * (loss_12 + loss_21)

  return jnp.mean(loss)


def simsiam_loss(
    predictor_projection_1: jax.typing.ArrayLike,
    target_projection_2: jax.typing.ArrayLike,
    predictor_projection_2: jax.typing.ArrayLike | None = None,
    target_projection_1: jax.typing.ArrayLike | None = None,
    eps: jax.typing.ArrayLike = 1e-6,
    symmetric: bool | None = None,
) -> jax.Array:
  """SimSiam loss.

  Computes the SimSiam negative cosine similarity loss between a predictor
  output and a target projection from two augmented views of the same inputs.

  The per-example loss is:

    D(p, z) = -cosine_similarity(p, stop_grad(z))

  Modes
  -----
  1. Single-direction:
     Uses only (p1, z2) and returns:
       mean(-cos(p1, stop_grad(z2)))

  2. Symmetric two-view:
     Uses both directions (p1 vs z2) and (p2 vs z1) and returns:
       mean(0.5 * [-cos(p1, stop_grad(z2))
                 - cos(p2, stop_grad(z1))])

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> p1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> z2 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> out = optax.losses.simsiam_loss(p1, z2)
    >>> print(out)
    -1.0

    >>> p2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    >>> z1 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    >>> out = optax.losses.simsiam_loss(p1, z2, p2, z1, symmetric=True)
    >>> print(out)
    -1.0

  Args:
    predictor_projection_1: Predictor output for view 1 (p1),
      shape [batch, feature_dim].
    target_projection_2: Target projection for view 2 (z2),
      shape [batch, feature_dim]. Treated as stop-gradient.
    predictor_projection_2: Optional predictor output for view 2 (p2),
      shape [batch, feature_dim]. Required when symmetric mode is enabled.
    target_projection_1: Optional target projection for view 1 (z1),
      shape [batch, feature_dim]. Treated as stop-gradient. Required when
      symmetric mode is enabled.
    eps: Small epsilon used inside cosine similarity for numerical stability.
    symmetric: If True, computes the symmetric SimSiam loss using both
      directions (p1 vs z2 and p2 vs z1). If False, or if left as None and the
      second pair is not provided, computes only the single-direction loss
      between `predictor_projection_1` and `target_projection_2`. If left as
      None and a second pair (`predictor_projection_2` / `target_projection_1`)
      is provided, symmetric mode is enabled automatically.

  Returns:
    A scalar SimSiam loss averaged over the batch. In symmetric mode, the loss
    is averaged over both directions and the batch.

  References:
    [1] X. Chen and K. He,
        `Exploring Simple Siamese Representation Learning
        <https://arxiv.org/abs/2011.10566>`_, 2021.
  """

  predictor_projection_1 = jnp.asarray(predictor_projection_1)
  target_projection_2 = jnp.asarray(target_projection_2)

  utils.check_subdtype(predictor_projection_1, jnp.floating)
  utils.check_subdtype(target_projection_2, jnp.floating)

  # Cast eps to match the projection dtype for numerical consistency.
  eps = jnp.asarray(eps, dtype=predictor_projection_1.dtype)

  # Decide symmetric mode in a jit-safe way.
  if symmetric is None or not isinstance(symmetric, bool):
    symmetric_flag = (
        predictor_projection_2 is not None or target_projection_1 is not None
    )
  else:
    symmetric_flag = symmetric

  # Single-direction mode: only (p1, z2) is used.
  if not symmetric_flag:
    if predictor_projection_1.shape != target_projection_2.shape:
      raise ValueError(
          'predictor_projection_1 and target_projection_2 must have the same '
          f'shape, found {predictor_projection_1.shape} and '
          f'{target_projection_2.shape}.'
      )

    # Stop gradient on target branch.
    target_projection_2 = lax.stop_gradient(target_projection_2)

    cos_12 = _regression.cosine_similarity(
        predictor_projection_1,
        target_projection_2,
        epsilon=eps,
    )
    loss_12 = -cos_12
    return jnp.mean(loss_12)

  # Symmetric mode: requires all four projections.
  if predictor_projection_2 is None or target_projection_1 is None:
    raise ValueError(
        'Symmetric SimSiam loss requested, but `predictor_projection_2` or '
        '`target_projection_1` is None. Provide all four projections or '
        'set `symmetric=False`.'
    )

  predictor_projection_2 = jnp.asarray(predictor_projection_2)
  target_projection_1 = jnp.asarray(target_projection_1)

  utils.check_subdtype(predictor_projection_2, jnp.floating)
  utils.check_subdtype(target_projection_1, jnp.floating)

  if predictor_projection_1.shape != target_projection_2.shape:
    raise ValueError(
        'predictor_projection_1 and target_projection_2 must have the same '
        f'shape, found {predictor_projection_1.shape} and '
        f'{target_projection_2.shape}.'
    )
  if predictor_projection_2.shape != target_projection_1.shape:
    raise ValueError(
        'predictor_projection_2 and target_projection_1 must have the same '
        f'shape, found {predictor_projection_2.shape} and '
        f'{target_projection_1.shape}.'
    )

  # Stop gradient on target branch.
  target_projection_1 = lax.stop_gradient(target_projection_1)
  target_projection_2 = lax.stop_gradient(target_projection_2)

  cos_12 = _regression.cosine_similarity(
      predictor_projection_1,
      target_projection_2,
      epsilon=eps,
  )
  cos_21 = _regression.cosine_similarity(
      predictor_projection_2,
      target_projection_1,
      epsilon=eps,
  )

  loss_12 = -cos_12
  loss_21 = -cos_21
  loss = 0.5 * (loss_12 + loss_21)

  return jnp.mean(loss)


def _dino_temperature_to_array(
    temperature: jax.typing.ArrayLike,
) -> jax.Array:
  """Convert temperature to array and ensure positivity for Python scalars."""
  if isinstance(temperature, (int, float)):
    if temperature <= 0:
      raise ValueError('Temperatures must be positive.')
  return jnp.asarray(temperature)


def _single_view_dino_loss(
    student_logits: jax.typing.ArrayLike,
    teacher_logits: jax.typing.ArrayLike,
    student_temperature: jax.Array,
    teacher_temperature: jax.Array,
    teacher_center: jax.Array,
) -> jax.Array:
  """Single-view DINO loss between a student and teacher distribution."""
  student_logits = jnp.asarray(student_logits)
  teacher_logits = jnp.asarray(teacher_logits)

  if student_logits.shape != teacher_logits.shape:
    raise ValueError(
        'student_logits and teacher_logits must have the same shape, found '
        f'{student_logits.shape} and {teacher_logits.shape}.'
    )

  # Apply centering and temperature scaling.
  teacher_scaled = (teacher_logits - teacher_center) / teacher_temperature
  student_scaled = student_logits / student_temperature

  # Probabilities from teacher (stop-gradient) and log-probs from student.
  teacher_prob = lax.stop_gradient(
      jax.nn.softmax(teacher_scaled, axis=-1)
  )
  log_student_prob = jax.nn.log_softmax(student_scaled, axis=-1)

  # Cross-entropy between teacher and student distributions.
  loss_per_example = -jnp.sum(teacher_prob * log_student_prob, axis=-1)
  return jnp.mean(loss_per_example)


def dino_loss(
    student_logits_1: jax.typing.ArrayLike,
    teacher_logits_1: jax.typing.ArrayLike,
    student_logits_2: jax.typing.ArrayLike | None = None,
    teacher_logits_2: jax.typing.ArrayLike | None = None,
    student_temperature: jax.typing.ArrayLike = 0.1,
    teacher_temperature: jax.typing.ArrayLike = 0.04,
    teacher_center: jax.typing.ArrayLike = 0.0,
    two_view: bool | None = None,
) -> jax.Array:
  """DINO loss (self-distillation with no labels).

  Computes the cross-entropy between teacher and student distributions as in
  DINO for self-supervised ViT training.

  The teacher distribution is obtained by applying centering and temperature
  scaling to the teacher logits, followed by a softmax. Gradients are stopped
  on the teacher branch. The student distribution is produced by applying a
  temperature-scaled log-softmax to the student logits.

  Modes
  -----
  1. Single-view:
     Matches student(x1) to teacher(x1) and returns:
       CE(stop_grad(softmax((t1 - c) / T_t)),
          log_softmax(s1 / T_s))

  2. Two-view symmetric:
     Matches student(x1) to teacher(x2) and student(x2) to teacher(x1), and
     averages the two directional losses:
       0.5 * [CE(t2 -> s1) + CE(t1 -> s2)]

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> s1 = jnp.zeros((2, 3))
    >>> t1 = jnp.zeros((2, 3))
    >>> out = optax.losses.dino_loss(s1, t1)

    >>> s2 = jnp.zeros((2, 3))
    >>> t2 = jnp.zeros((2, 3))
    >>> out = optax.losses.dino_loss(s1, t1, s2, t2, two_view=True)

  Args:
    student_logits_1: Logits from the student network for view 1,
      shape [..., num_classes].
    teacher_logits_1: Logits from the teacher network for view 1,
      shape [..., num_classes]. Treated as stop-gradient after softmax.
    student_logits_2: Optional logits from the student network for view 2,
      shape [..., num_classes]. Required when `two_view=True`.
    teacher_logits_2: Optional logits from the teacher network for view 2,
      shape [..., num_classes]. Required when `two_view=True`.
    student_temperature: Temperature for the student softmax. Must be positive.
    teacher_temperature: Temperature for the teacher softmax. Must be positive.
    teacher_center: Centering term added to teacher logits before temperature
      scaling. May be a scalar or an array broadcastable to the shape of
      `teacher_logits_*`.
    two_view: If True, computes the two-view symmetric DINO loss using both
      directions (student_1 vs teacher_2 and student_2 vs teacher_1). If False,
      or if left as None and the second pair is not provided, computes only the
      single-view loss between `student_logits_1` and `teacher_logits_1`. If
      left as None and a second pair (`student_logits_2` / `teacher_logits_2`)
      is provided, two-view symmetric mode is enabled automatically.

  Returns:
    A scalar DINO loss. In single-view mode, it is the mean cross-entropy over
    the batch. In two-view mode, it is the average of the two directional
    losses, each already averaged over the batch.

  References:
    [1] M. Caron et al,
        `Emerging Properties in Self-Supervised Vision Transformers
        <https://arxiv.org/abs/2104.14294>`_, 2021.
  """

  # Convert logits and center to arrays.
  student_logits_1 = jnp.asarray(student_logits_1)
  teacher_logits_1 = jnp.asarray(teacher_logits_1)
  if student_logits_2 is not None:
    student_logits_2 = jnp.asarray(student_logits_2)
  if teacher_logits_2 is not None:
    teacher_logits_2 = jnp.asarray(teacher_logits_2)
  teacher_center = jnp.asarray(teacher_center)

  utils.check_subdtype(student_logits_1, jnp.floating)
  utils.check_subdtype(teacher_logits_1, jnp.floating)
  if student_logits_2 is not None:
    utils.check_subdtype(student_logits_2, jnp.floating)
  if teacher_logits_2 is not None:
    utils.check_subdtype(teacher_logits_2, jnp.floating)

  # Ensure temperatures are positive (for Python scalars) and convert to arrays.
  student_temperature = _dino_temperature_to_array(student_temperature)
  teacher_temperature = _dino_temperature_to_array(teacher_temperature)

  # Decide two-view mode in a jit-safe way.
  if two_view is None or not isinstance(two_view, bool):
    two_view_flag = (
        student_logits_2 is not None or teacher_logits_2 is not None
    )
  else:
    two_view_flag = two_view

  # Single-view mode: use only (student_logits_1, teacher_logits_1).
  if not two_view_flag:
    return _single_view_dino_loss(
        student_logits_1,
        teacher_logits_1,
        student_temperature,
        teacher_temperature,
        teacher_center,
    )

  # Two-view mode: requires all four logits.
  if student_logits_2 is None or teacher_logits_2 is None:
    raise ValueError(
        'Two-view DINO loss requested, but `student_logits_2` or '
        '`teacher_logits_2` is None. Provide all four logits or '
        'set `two_view=False`.'
    )

  if not (
      student_logits_1.shape
      == student_logits_2.shape
      == teacher_logits_1.shape
      == teacher_logits_2.shape
  ):
    raise ValueError(
        'In two-view mode, all logits must have the same shape. Found '
        f'student_logits_1: {student_logits_1.shape}, '
        f'student_logits_2: {student_logits_2.shape}, '
        f'teacher_logits_1: {teacher_logits_1.shape}, '
        f'teacher_logits_2: {teacher_logits_2.shape}.'
    )

  loss_12 = _single_view_dino_loss(
      student_logits_1,
      teacher_logits_2,
      student_temperature,
      teacher_temperature,
      teacher_center,
  )
  loss_21 = _single_view_dino_loss(
      student_logits_2,
      teacher_logits_1,
      student_temperature,
      teacher_temperature,
      teacher_center,
  )

  return 0.5 * (loss_12 + loss_21)


def barlow_twins_loss(
    projection_1: jax.typing.ArrayLike,
    projection_2: jax.typing.ArrayLike,
    off_diagonal_scale: jax.typing.ArrayLike = 5e-3,
    eps: jax.typing.ArrayLike = 1e-12,
) -> jax.Array:
  """Barlow Twins loss.

  Computes the Barlow Twins redundancy reduction loss between two batches of
  projections corresponding to two augmented views of the same inputs.

  Given two batches of projections z1 and z2 with shape [batch, feature_dim],
  Barlow Twins computes the cross-correlation matrix between features of the
  two views and encourages:
    * On-diagonal entries C_ii to be close to 1.
    * Off-diagonal entries C_ij (i != j) to be close to 0.

  The loss is:
      L = sum_i (1 - C_ii)^2 + lambda * sum_{i != j} C_ij^2

  where lambda is `off_diagonal_scale`.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> z1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> z2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> out = optax.losses.barlow_twins_loss(z1, z2)

  Args:
    projection_1: Projections for view 1, shape [batch, feature_dim].
    projection_2: Projections for view 2, shape [batch, feature_dim].
    off_diagonal_scale: Weight for off-diagonal correlation terms.
    eps: Small epsilon added to variances for numerical stability.

  Returns:
    A scalar Barlow Twins loss.

  References:
    [1] J. Zbontar et al,
        `Barlow Twins: Self-Supervised Learning via Redundancy Reduction
        <https://arxiv.org/abs/2103.03230>`_, 2021.
  """

  projection_1 = jnp.asarray(projection_1)
  projection_2 = jnp.asarray(projection_2)

  utils.check_subdtype(projection_1, jnp.floating)
  utils.check_subdtype(projection_2, jnp.floating)

  if projection_1.shape != projection_2.shape:
    raise ValueError(
        'projection_1 and projection_2 must have the same shape, found '
        f'{projection_1.shape} and {projection_2.shape}.'
    )
  if projection_1.ndim != 2:
    raise ValueError(
        'Barlow Twins expects rank-2 inputs [batch, feature_dim], found '
        f'shape {projection_1.shape}.'
    )

  batch_size, _ = projection_1.shape

  eps = jnp.asarray(eps, dtype=projection_1.dtype)
  off_diagonal_scale = jnp.asarray(
      off_diagonal_scale, dtype=projection_1.dtype
  )

  # Normalize each feature dimension across the batch.
  proj1_mean = jnp.mean(projection_1, axis=0, keepdims=True)
  proj2_mean = jnp.mean(projection_2, axis=0, keepdims=True)
  proj1_centered = projection_1 - proj1_mean
  proj2_centered = projection_2 - proj2_mean

  proj1_var = jnp.mean(proj1_centered ** 2, axis=0, keepdims=True)
  proj2_var = jnp.mean(proj2_centered ** 2, axis=0, keepdims=True)
  proj1_std = jnp.sqrt(proj1_var + eps)
  proj2_std = jnp.sqrt(proj2_var + eps)

  proj1_norm = proj1_centered / proj1_std
  proj2_norm = proj2_centered / proj2_std

  # Cross-correlation matrix C_ij.
  cross_correlation = (proj1_norm.T @ proj2_norm) / batch_size

  on_diag = jnp.diag(cross_correlation)
  on_diag_loss = jnp.sum((1.0 - on_diag) ** 2)

  corr_sq = cross_correlation ** 2
  off_diag_loss = (
      jnp.sum(corr_sq) - jnp.sum(jnp.diag(corr_sq))
  )

  loss = on_diag_loss + off_diagonal_scale * off_diag_loss
  return loss
