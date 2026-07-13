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

from typing import Optional

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
  # pyrefly: ignore [missing-attribute]
  if labels.shape[0] != embeddings.shape[0]:  # pytype: disable=attribute-error  # jax-arraylike # noqa: E501
    raise ValueError(
        'Labels and embeddings must have the same leading dimension, found'
        # pyrefly: ignore [missing-attribute]
        f' {labels.shape[0]} for labels and {embeddings.shape[0]} for'  # pytype: disable=attribute-error  # jax-arraylike # noqa: E501
        ' embeddings.'
    )

  # cosine similarity matrix
  xcs = (
      _regression.cosine_similarity(
          embeddings[None, :, :],  # pyrefly: ignore[bad-index]
          embeddings[:, None, :],  # pyrefly: ignore[bad-index]
          # pyrefly: ignore [missing-attribute]
          epsilon=jnp.finfo(embeddings.dtype).eps,  # pytype: disable=attribute-error  # jax-arraylike # noqa: E501
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


def _check_optional_pair(
    first: Optional[jax.typing.ArrayLike],
    second: Optional[jax.typing.ArrayLike],
    first_name: str,
    second_name: str,
) -> bool:
  """Returns whether an optional pair is present, validating completeness."""
  if (first is None) != (second is None):
    raise ValueError(
        f'`{first_name}` and `{second_name}` must either both be provided or '
        'both be None.'
    )
  return first is not None


def _check_same_shapes(
    reference: jax.typing.ArrayLike,
    reference_name: str,
    *others: tuple[jax.typing.ArrayLike, str],
):
  """Checks that all arrays have the same shape as a reference array."""
  reference_shape = jnp.shape(reference)
  for array, name in others:
    array_shape = jnp.shape(array)
    if reference_shape != array_shape:
      raise ValueError(
          f'`{reference_name}` and `{name}` must have the same shape, found '
          f'{reference_shape} and {array_shape}.'
      )


def _negative_cosine_similarity(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    eps: jax.typing.ArrayLike,
) -> jax.Array:
  """Computes negative cosine similarity with stopped target gradients."""
  targets = lax.stop_gradient(targets)
  return -_regression.cosine_similarity(
      predictions, targets, epsilon=eps, axis=-1
  )


def byol_loss(
    online_prediction_1: jax.typing.ArrayLike,
    target_projection_2: jax.typing.ArrayLike,
    online_prediction_2: Optional[jax.typing.ArrayLike] = None,
    target_projection_1: Optional[jax.typing.ArrayLike] = None,
    *,
    eps: jax.typing.ArrayLike = 1e-6,
) -> jax.Array:
  r"""Computes the Bootstrap Your Own Latent (BYOL) loss.

  BYOL regresses online-network predictions toward target-network projections
  computed from another augmented view of the same examples. Targets are
  treated as stop-gradient values inside this loss.

  For one direction, with online prediction :math:`q` and target projection
  :math:`z`, this function computes the squared distance between
  :math:`\ell_2`-normalized vectors,

  .. math::
    D(q, z) = \|\bar{q} - \bar{z}\|_2^2 = 2 - 2\cos(q, z).

  If `online_prediction_2` and `target_projection_1` are both provided, the
  function returns the symmetric two-view BYOL objective:

  .. math::
    \frac{1}{2}\left(D(q_1, z_2) + D(q_2, z_1)\right).

  .. note::
    The BYOL paper minimizes the *sum* of the two directions,
    :math:`D(q_1, z_2) + D(q_2, z_1)`. This function averages them instead,
    for consistency with :func:`simsiam_loss
    <optax.losses.simsiam_loss>`; multiply the result by two to recover the
    paper's objective.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> online_prediction = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> target_projection = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    >>> print(optax.losses.byol_loss(online_prediction, target_projection))
    [2. 2.]
    >>> print(optax.losses.byol_loss(online_prediction, online_prediction))
    [0. 0.]

  Args:
    online_prediction_1: Online-network prediction for view 1, with shape
      `[..., feature_dim]`.
    target_projection_2: Target-network projection for view 2, with the same
      shape as `online_prediction_1`. Gradients are stopped through this
      argument.
    online_prediction_2: Optional online-network prediction for view 2. If
      provided, `target_projection_1` must also be provided.
    target_projection_1: Optional target-network projection for view 1.
      Gradients are stopped through this argument. If provided,
      `online_prediction_2` must also be provided.
    eps: Minimum squared norm enforced in the cosine-similarity denominator,
      so the effective minimum norm is `sqrt(eps)`.

  Returns:
    BYOL loss values for each example, with shape `[...]`. When both views
    are provided, each value averages the two directions. Take the mean for
    a scalar batch loss.

  References:
    Grill et al, `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_, 2020.

  .. versionadded:: 0.2.9
  """
  online_prediction_1 = jnp.asarray(online_prediction_1)
  target_projection_2 = jnp.asarray(target_projection_2)
  utils.check_subdtype(online_prediction_1, jnp.floating)
  utils.check_subdtype(target_projection_2, jnp.floating)
  _check_same_shapes(
      online_prediction_1,
      'online_prediction_1',
      (target_projection_2, 'target_projection_2'),
  )

  eps = jnp.asarray(eps, dtype=online_prediction_1.dtype)
  loss_12 = 2.0 + 2.0 * _negative_cosine_similarity(
      online_prediction_1, target_projection_2, eps
  )

  if not _check_optional_pair(
      online_prediction_2,
      target_projection_1,
      'online_prediction_2',
      'target_projection_1',
  ):
    return loss_12

  online_prediction_2 = jnp.asarray(online_prediction_2)
  target_projection_1 = jnp.asarray(target_projection_1)
  utils.check_subdtype(online_prediction_2, jnp.floating)
  utils.check_subdtype(target_projection_1, jnp.floating)
  _check_same_shapes(
      online_prediction_1,
      'online_prediction_1',
      (online_prediction_2, 'online_prediction_2'),
      (target_projection_1, 'target_projection_1'),
  )

  loss_21 = 2.0 + 2.0 * _negative_cosine_similarity(
      online_prediction_2, target_projection_1, eps
  )
  return 0.5 * (loss_12 + loss_21)


def simsiam_loss(
    prediction_1: jax.typing.ArrayLike,
    target_projection_2: jax.typing.ArrayLike,
    prediction_2: Optional[jax.typing.ArrayLike] = None,
    target_projection_1: Optional[jax.typing.ArrayLike] = None,
    *,
    eps: jax.typing.ArrayLike = 1e-6,
) -> jax.Array:
  r"""Computes the SimSiam negative cosine similarity loss.

  SimSiam compares a predictor output (prediction) computed from one
  augmented view with a stop-gradient projection computed from another view:

  .. math::
    D(p, z) = -\cos(p, \operatorname{stop\_gradient}(z)).

  If `prediction_2` and `target_projection_1` are both provided, the function
  returns the symmetric two-view objective of the paper:

  .. math::
    \frac{1}{2} D(p_1, z_2) + \frac{1}{2} D(p_2, z_1).

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> prediction = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    >>> print(optax.losses.simsiam_loss(prediction, prediction))
    [-1. -1.]

  Args:
    prediction_1: Predictor output for view 1, with shape
      `[..., feature_dim]`.
    target_projection_2: Projection for view 2, used as the regression
      target, with the same shape as `prediction_1`. Gradients are stopped
      through this argument.
    prediction_2: Optional predictor output for view 2. If provided,
      `target_projection_1` must also be provided.
    target_projection_1: Optional projection for view 1, used as the
      regression target. Gradients are stopped through this argument. If
      provided, `prediction_2` must also be provided.
    eps: Minimum squared norm enforced in the cosine-similarity denominator,
      so the effective minimum norm is `sqrt(eps)`.

  Returns:
    SimSiam loss values for each example, with shape `[...]`. When both views
    are provided, each value averages the two directions. Take the mean for a
    scalar batch loss.

  References:
    Chen and He, `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_, 2021.

  .. versionadded:: 0.2.9
  """
  prediction_1 = jnp.asarray(prediction_1)
  target_projection_2 = jnp.asarray(target_projection_2)
  utils.check_subdtype(prediction_1, jnp.floating)
  utils.check_subdtype(target_projection_2, jnp.floating)
  _check_same_shapes(
      prediction_1,
      'prediction_1',
      (target_projection_2, 'target_projection_2'),
  )

  eps = jnp.asarray(eps, dtype=prediction_1.dtype)
  loss_12 = _negative_cosine_similarity(
      prediction_1, target_projection_2, eps
  )

  if not _check_optional_pair(
      prediction_2,
      target_projection_1,
      'prediction_2',
      'target_projection_1',
  ):
    return loss_12

  prediction_2 = jnp.asarray(prediction_2)
  target_projection_1 = jnp.asarray(target_projection_1)
  utils.check_subdtype(prediction_2, jnp.floating)
  utils.check_subdtype(target_projection_1, jnp.floating)
  _check_same_shapes(
      prediction_1,
      'prediction_1',
      (prediction_2, 'prediction_2'),
      (target_projection_1, 'target_projection_1'),
  )

  loss_21 = _negative_cosine_similarity(
      prediction_2, target_projection_1, eps
  )
  return 0.5 * (loss_12 + loss_21)


def _positive_temperature(
    temperature: jax.typing.ArrayLike, name: str
) -> jax.Array:
  """Validates that a temperature is a scalar and, when concrete, positive."""
  temperature = jnp.asarray(temperature)
  if jnp.ndim(temperature) != 0:
    raise ValueError(f'`{name}` must be a scalar.')
  try:
    is_positive = bool(temperature > 0)
  except jax.errors.TracerBoolConversionError:
    # Traced temperatures cannot be validated at trace time.
    return temperature
  if not is_positive:
    raise ValueError(f'`{name}` must be positive.')
  return temperature


def _single_view_dino_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    student_temperature: jax.Array,
    teacher_temperature: jax.Array,
    teacher_center: jax.typing.ArrayLike,
) -> jax.Array:
  """Computes one DINO teacher-to-student cross entropy term per example."""
  teacher_center = jnp.asarray(teacher_center)
  utils.check_subdtype(teacher_center, jnp.floating)
  teacher_center = teacher_center.astype(teacher_logits.dtype)
  student_temperature = student_temperature.astype(student_logits.dtype)
  teacher_temperature = teacher_temperature.astype(teacher_logits.dtype)
  logits_shape = jnp.shape(teacher_logits)
  center_error = ValueError(
      '`teacher_center` must be broadcastable to the teacher logits shape, '
      f'found {jnp.shape(teacher_center)} and {logits_shape}.'
  )
  try:
    broadcast_shape = jnp.broadcast_shapes(
        jnp.shape(teacher_center), logits_shape
    )
  except ValueError as error:
    raise center_error from error
  if broadcast_shape != logits_shape:
    raise center_error

  teacher_probs = lax.stop_gradient(
      jax.nn.softmax(
          (teacher_logits - teacher_center) / teacher_temperature, axis=-1
      )
  )
  student_log_probs = jax.nn.log_softmax(
      student_logits / student_temperature, axis=-1
  )
  return -jnp.sum(teacher_probs * student_log_probs, axis=-1)


def dino_loss(
    student_logits_1: jax.typing.ArrayLike,
    teacher_logits_2: jax.typing.ArrayLike,
    student_logits_2: Optional[jax.typing.ArrayLike] = None,
    teacher_logits_1: Optional[jax.typing.ArrayLike] = None,
    *,
    student_temperature: jax.typing.ArrayLike = 0.1,
    teacher_temperature: jax.typing.ArrayLike = 0.04,
    teacher_center: jax.typing.ArrayLike = 0.0,
) -> jax.Array:
  r"""Computes the DINO self-distillation loss.

  DINO trains a student distribution to match a centered and sharpened
  teacher distribution computed from a different augmented view of the same
  examples. The teacher branch is treated as stop-gradient.

  For one (student, teacher) pair of views, this function computes the cross
  entropy

  .. math::
    -\sum_k p_t(k)\log p_s(k),

  where :math:`p_t` is `softmax((teacher_logits - teacher_center) /
  teacher_temperature)` and :math:`p_s` is `softmax(student_logits /
  student_temperature)`.

  If `student_logits_2` and `teacher_logits_1` are both provided, the
  function returns the two-view specialization of the DINO multi-crop
  objective, averaging the term for `student_logits_1` against
  `teacher_logits_2` with the term for `student_logits_2` against
  `teacher_logits_1`. DINO never compares student and teacher outputs of the
  same view, which is why the argument indices are cross-paired, as in
  :func:`byol_loss <optax.losses.byol_loss>`. For more than two crops, call
  the single-pair form once per (student view, teacher view) pair of
  distinct views and average the results.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> student_logits = jnp.zeros((2, 4))
    >>> teacher_logits = jnp.zeros((2, 4))
    >>> loss = optax.losses.dino_loss(student_logits, teacher_logits)
    >>> print(loss)  # cross entropy of uniform distributions, i.e. log(4)
    [1.3862944 1.3862944]

  Args:
    student_logits_1: Student logits for view 1, with shape
      `[..., num_classes]`.
    teacher_logits_2: Teacher logits for view 2, with the same shape as
      `student_logits_1`. Gradients are stopped through this argument.
    student_logits_2: Optional student logits for view 2. If provided,
      `teacher_logits_1` must also be provided.
    teacher_logits_1: Optional teacher logits for view 1. Gradients are
      stopped through this argument. If provided, `student_logits_2` must
      also be provided.
    student_temperature: Positive temperature for the student softmax. The
      paper uses 0.1. Positivity can only be validated for concrete
      (non-traced) values.
    teacher_temperature: Positive temperature for the teacher softmax. The
      paper warms this up from 0.04 to 0.07; pass the current schedule value.
      Positivity can only be validated for concrete (non-traced) values.
    teacher_center: Centering term subtracted from teacher logits before
      temperature scaling. Must be broadcastable to the teacher logits shape,
      and is cast to the teacher logits dtype. The paper maintains this
      center as an exponential moving average of teacher outputs across
      batches; the caller is responsible for updating it between steps. The
      default 0.0 applies no centering.

  Returns:
    DINO cross-entropy loss values for each example, with shape `[...]`.
    When both views are provided, each value averages the two cross-view
    terms. Take the mean for a scalar batch loss.

  References:
    Caron et al, `Emerging Properties in Self-Supervised Vision Transformers
    <https://arxiv.org/abs/2104.14294>`_, 2021.

  .. versionadded:: 0.2.9
  """
  student_temperature = _positive_temperature(
      student_temperature, 'student_temperature'
  )
  teacher_temperature = _positive_temperature(
      teacher_temperature, 'teacher_temperature'
  )

  student_logits_1 = jnp.asarray(student_logits_1)
  teacher_logits_2 = jnp.asarray(teacher_logits_2)
  utils.check_subdtype(student_logits_1, jnp.floating)
  utils.check_subdtype(teacher_logits_2, jnp.floating)
  _check_same_shapes(
      student_logits_1,
      'student_logits_1',
      (teacher_logits_2, 'teacher_logits_2'),
  )

  loss_12 = _single_view_dino_loss(
      student_logits_1,
      teacher_logits_2,
      student_temperature,
      teacher_temperature,
      teacher_center,
  )

  if not _check_optional_pair(
      student_logits_2, teacher_logits_1, 'student_logits_2', 'teacher_logits_1'
  ):
    return loss_12

  student_logits_2 = jnp.asarray(student_logits_2)
  teacher_logits_1 = jnp.asarray(teacher_logits_1)
  utils.check_subdtype(student_logits_2, jnp.floating)
  utils.check_subdtype(teacher_logits_1, jnp.floating)
  _check_same_shapes(
      student_logits_1,
      'student_logits_1',
      (student_logits_2, 'student_logits_2'),
      (teacher_logits_1, 'teacher_logits_1'),
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
    *,
    off_diagonal_scale: jax.typing.ArrayLike = 5e-3,
    eps: jax.typing.ArrayLike = 1e-5,
) -> jax.Array:
  r"""Computes the Barlow Twins redundancy-reduction loss.

  Barlow Twins compares two batches of projections for paired augmented views.
  Each feature is standardized across the batch dimension (using the biased,
  `1/batch_size`, variance, as in the batch-normalization-based reference
  implementation), then a feature cross-correlation matrix :math:`C` is
  computed. The objective pushes diagonal entries toward one and off-diagonal
  entries toward zero:

  .. math::
    \sum_i (1 - C_{ii})^2
    + \lambda \sum_i \sum_{j \ne i} C_{ij}^2.

  This loss couples examples through batch statistics, so it returns a single
  scalar and requires at least two examples.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> projections = jnp.array([[1.0, 1.0, 1.0],
    ...                          [-1.0, 1.0, -1.0],
    ...                          [1.0, -1.0, -1.0],
    ...                          [-1.0, -1.0, 1.0]])
    >>> loss = optax.losses.barlow_twins_loss(projections, projections)
    >>> print(f'{loss:.4f}')
    0.0000

  Args:
    projection_1: Rank-2 array of projections for view 1, with shape
      `[batch_size, feature_dim]` and `batch_size >= 2`.
    projection_2: Rank-2 array of projections for view 2, with the same shape
      as `projection_1`.
    off_diagonal_scale: Multiplicative scale :math:`\lambda` for
      off-diagonal terms.
    eps: Small value added to per-feature variances before taking square
      roots. The default matches the batch-normalization epsilon used by the
      reference implementation.

  Returns:
    Scalar Barlow Twins loss.

  References:
    Zbontar et al, `Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction <https://arxiv.org/abs/2103.03230>`_, 2021.

  .. versionadded:: 0.2.9
  """
  projection_1 = jnp.asarray(projection_1)
  projection_2 = jnp.asarray(projection_2)
  utils.check_subdtype(projection_1, jnp.floating)
  utils.check_subdtype(projection_2, jnp.floating)
  _check_same_shapes(
      projection_1, 'projection_1', (projection_2, 'projection_2')
  )
  utils.check_rank(projection_1, 2)

  batch_size = projection_1.shape[0]
  if batch_size < 2:
    raise ValueError(
        '`projection_1` and `projection_2` must contain at least two examples'
        f' to compute per-feature statistics, found batch size {batch_size}.'
    )

  eps = jnp.asarray(eps, dtype=projection_1.dtype)
  off_diagonal_scale = jnp.asarray(off_diagonal_scale, dtype=projection_1.dtype)

  projection_1 = projection_1 - jnp.mean(projection_1, axis=0, keepdims=True)
  projection_2 = projection_2 - jnp.mean(projection_2, axis=0, keepdims=True)
  projection_1 = projection_1 / jnp.sqrt(
      jnp.mean(projection_1**2, axis=0, keepdims=True) + eps
  )
  projection_2 = projection_2 / jnp.sqrt(
      jnp.mean(projection_2**2, axis=0, keepdims=True) + eps
  )

  cross_correlation = (projection_1.T @ projection_2) / batch_size
  on_diagonal = jnp.diag(cross_correlation)
  on_diagonal_loss = jnp.sum((1.0 - on_diagonal) ** 2)

  off_diagonal_loss = jnp.maximum(
      0.0, jnp.sum(cross_correlation**2) - jnp.sum(on_diagonal**2)
  )
  return on_diagonal_loss + off_diagonal_scale * off_diagonal_loss
