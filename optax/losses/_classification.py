# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Classification losses."""

import functools
import operator
from typing import Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.special
from optax import projections
from optax._src import utils
from optax._src.deprecations import warn_deprecated_function  # pylint: disable=g-importing-member


def canonicalize_axis(axis, ndim):
  """Vendored version of :func:`numpy.lib.array_utils.normalize_axis_index`.
  """
  if 0 <= (axis := operator.index(axis)) < ndim:
    return axis
  elif -ndim <= axis < 0:
    return axis + ndim
  else:
    raise ValueError(f'axis {axis} is out of bounds for array of '
                     f'dimension {ndim}')


def canonicalize_axes(axes, ndim) -> tuple[int, ...]:
  """Vendored version of :func:`numpy.lib.array_utils.normalize_axis_tuple`.
  """
  return tuple(canonicalize_axis(x, ndim) for x in axes)


def sigmoid_binary_cross_entropy(
    logits,
    labels,
):
  """Computes element-wise sigmoid cross entropy given logits and labels.

  This function can be used for binary or multiclass classification (where each
  class is an independent binary prediction and different classes are not
  mutually exclusive e.g. predicting that an image contains both a cat
  and a dog.)

  Because this function is overloaded, please ensure your `logits` and `labels`
  are compatible with each other. If you're passing in binary `labels` (values
  in {0, 1}), ensure your `logits` correspond to class 1 only. If you're
  passing in per-class target probabilities or one-hot `labels`, please ensure
  your `logits` are also multiclass. Be particularly careful if you're relying
  on implicit broadcasting to reshape `logits` or `labels`.

  Args:
    logits: Each element is the unnormalized log probability of a binary
      prediction. See note about compatibility with `labels` above.
    labels: Binary labels whose values are {0,1} or multi-class target
      probabilities. See note about compatibility with `logits` above.

  Returns:
    cross entropy for each binary prediction, same shape as `logits`.

  References:
    Goodfellow et al, `Deep Learning
    <http://www.deeplearningbook.org/contents/prob.html>`_, 2016
  """
  utils.check_subdtype(logits, jnp.floating)
  labels = jnp.astype(labels, logits.dtype)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1.0 - labels) * log_not_p


@functools.partial(
    warn_deprecated_function, replacement='sigmoid_binary_cross_entropy'
)
def binary_logistic_loss(logits, labels):
  return sigmoid_binary_cross_entropy(logits, labels)


def hinge_loss(
    predictor_outputs: jax.typing.ArrayLike, targets: jax.typing.ArrayLike
) -> jax.Array:
  """Computes the hinge loss for binary classification.

  Args:
    predictor_outputs: Outputs of the decision function.
    targets: Target values. Target values should be strictly in the set {-1, 1}.

  Returns:
    loss value.
  """
  return jnp.maximum(0, 1 - predictor_outputs * targets)


def perceptron_loss(
    predictor_outputs: jax.typing.ArrayLike, targets: jax.typing.ArrayLike
) -> jax.Array:
  """Binary perceptron loss.

  Args:
    predictor_outputs: score produced by the model (float).
    targets: Target values. Target values should be strictly in the set {-1, 1}.

  Returns:
    loss value.

  References:
    `Perceptron <https://en.wikipedia.org/wiki/Perceptron>`_, Wikipedia
  """
  utils.check_shapes_equal(predictor_outputs, targets)
  return jnp.maximum(0, -predictor_outputs * targets)


def sparsemax_loss(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
) -> jax.Array:
  """Binary sparsemax loss.

  This loss is zero if and only if `jax.nn.sparse_sigmoid(logits) == labels`.

  Args:
    logits: score produced by the model (float).
    labels: ground-truth integer label (0 or 1).

  Returns:
    loss value

  References:
    Learning with Fenchel-Young Losses. Mathieu Blondel, André F. T. Martins,
    Vlad Niculae. JMLR 2020. (Sec. 4.4)

  .. versionadded:: 0.2.3
  """
  return jax.nn.sparse_plus(jnp.where(labels, -logits, logits))


@functools.partial(warn_deprecated_function, replacement='sparsemax_loss')
def binary_sparsemax_loss(logits, labels):
  return sparsemax_loss(logits, labels)


@jax.custom_jvp
def weighted_logsoftmax(
    x: jax.typing.ArrayLike, weights: jax.typing.ArrayLike) -> jax.Array:
  r"""Weighted logsoftmax.

  Computes
  .. math::
    (w_i \log(\exp x_i /(\sum_i \exp x_i )) )_{i=1}^n

  for :math:`x` the input ``x``, :math:`w` the ``weights``.
  For :math:`w_i = 0`, :math:`x_i=-\infty`, this implementation ensures that the
  output is 0 and not nan at the ith entry following the convention that
  :math:`0 \log 0 = 0`.

  Args:
    x: input array.
    weights: weights.

  Returns:
    logsoftmax of x multiplied elementwise by weights
  """
  logsoftmax_x = jax.nn.log_softmax(x, axis=-1)
  return jnp.where(
      weights != 0.0, weights * logsoftmax_x, jnp.zeros_like(logsoftmax_x)
  )


def _weighted_logsoftmax_jvp(primals, tangents):
  """Custom JVP of weighted logsoftmax."""
  (x, weights) = primals
  (x_dot, weights_dot) = tangents
  logsoftmax_x = jax.nn.log_softmax(x, axis=-1)
  result = jnp.where(
      weights != 0.0, weights * logsoftmax_x, jnp.zeros_like(logsoftmax_x)
  )
  out_tangents = (
      weights * x_dot
      - weights
      * jnp.sum(x_dot * jax.nn.softmax(x, axis=-1), axis=-1, keepdims=True)
      + weights_dot * logsoftmax_x
  )
  return result, out_tangents


weighted_logsoftmax.defjvp(_weighted_logsoftmax_jvp)


def safe_softmax_cross_entropy(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
) -> jax.Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  Contrarily to :func:`optax.softmax_cross_entropy` this function handles
  ``labels*logsoftmax(logits)`` as ``0`` when ``logits=-inf`` and ``labels=0``,
  following the convention that ``0 log 0 = 0``.

  Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Valid probability distributions (non-negative, sum to 1), e.g a one
      hot encoding specifying the correct class for each input; must have a
      shape broadcastable to `[..., num_classes]`.

  Returns:
    cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
  """
  utils.check_subdtype(logits, jnp.floating)
  return -jnp.sum(weighted_logsoftmax(logits, labels), axis=-1)


def softmax_cross_entropy(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  r"""Computes the softmax cross entropy between sets of logits and labels.

  This loss function is commonly used for multi-class classification tasks. It
  measures the dissimilarity between the predicted probability distribution
  (obtained by applying the softmax function to the logits) and the true
  probability distribution (represented by the one-hot encoded labels).
  This loss is also known as categorical cross entropy.

  Let :math:`x` denote the ``logits`` array of size ``[batch_size,
  num_classes]`` and :math:`y` denote the ``labels`` array of size
  ``[batch_size, num_classes]``. Then this function returns a vector
  :math:`\sigma` of size ``[batch_size]`` defined as:

  .. math::
    \sigma_i =
    - \sum_j y_{i j} \log\left(\frac{\exp(x_{i j})}{\sum_k
    \exp(x_{i k})}\right) \,.

  Args:
    logits: Unnormalized log probabilities, with shape ``[batch_size,
      num_classes]``.
    labels: One-hot encoded labels, with shape `[batch_size, num_classes]`. Each
      row represents the true class distribution for a single example.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation of shape ``[batch_size]`` or
      logits.shape.

  Returns:
    Cross-entropy between each prediction and the corresponding target
    distributions, with shape ``[batch_size]``.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> jnp.set_printoptions(precision=4)
    >>> # example: batch_size = 2, num_classes = 3
    >>> logits = jnp.array([[1.2, -0.8, -0.5], [0.9, -1.2, 1.1]])
    >>> labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> print(optax.softmax_cross_entropy(logits, labels))
    [0.2761 2.9518]

  References:
    `Cross-entropy Loss <https://en.wikipedia.org/wiki/Cross-entropy>`_,
    Wikipedia

    `Multinomial Logistic Regression
    <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_, Wikipedia

  .. seealso::
    This function is similar to
    :func:`optax.losses.softmax_cross_entropy_with_integer_labels`,
    but accepts one-hot labels instead of integer labels.

    :func:`optax.losses.safe_softmax_cross_entropy` provides an alternative
    implementation that differs on how ``logits=-inf`` are handled.

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(logits, jnp.floating)
  if where is not None and where.ndim != logits.ndim:
    where = jnp.expand_dims(where, axis)
  log_probs = jax.nn.log_softmax(logits, axis, where)
  return -(labels * log_probs).sum(axis, where=where)


def softmax_cross_entropy_with_integer_labels(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...]] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  r"""Computes softmax cross entropy between the logits and integer labels.

  This loss is useful for classification problems with integer labels that are
  not one-hot encoded. This loss is also known as categorical cross entropy.

  Let :math:`x` denote the ``logits`` array of size ``[batch_size,
  num_classes]`` and :math:`y` denote the ``labels`` array of size
  ``[batch_size]``. Then this function returns a vector
  :math:`\sigma` of size ``[batch_size]`` defined as:

  .. math::
    \sigma_i =
    \log\left(\frac{\exp(x_{i y_i})}{\sum_j
    \exp(x_{i j})}\right)\,.

  Args:
    logits: Unnormalized log probabilities, with shape ``[batch_size,
      num_classes]``.
    labels: Integers specifying the correct class for each input, with shape
      ``[batch_size]``. Class labels are assumed to be between 0 and
      ``num_classes - 1`` inclusive.
    axis: Axis or axes along which to compute. If a tuple of axes is passed
      then ``num_classes`` must match the total number of elements in ``axis``
      dimensions and a label is interpreted as a flat index in a ``logits``
      slice of shape ``logits[axis]``.
    where: Elements to include in the computation of shape ``[batch_size]``
      or logits.shape.

  Returns:
    Cross-entropy between each prediction and the corresponding target
    distributions, with shape ``[batch_size]``.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> jnp.set_printoptions(precision=4)
    >>> # example: batch_size = 2, num_classes = 3
    >>> logits = jnp.array([[1.2, -0.8, -0.5], [0.9, -1.2, 1.1]])
    >>> labels = jnp.array([0, 1])
    >>> print(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    [0.2761 2.9518]

    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import optax
    >>> jnp.set_printoptions(precision=4)
    >>> # example: batch_size = (1, 2), num_classes = 12 (i.e. 3 * 4)
    >>> shape = (1, 2, 3, 4)
    >>> logits = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    >>> # elements indices in slice of shape (3, 4)
    >>> ix = jnp.array([[1, 2]])
    >>> jx = jnp.array([[1, 3]])
    >>> labels = jnp.ravel_multi_index((ix, jx), shape[2:])
    >>> cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
    ...     logits, labels, axis=(2, 3))
    >>> print(cross_entropy)
    [[6.4587 0.4587]]

  References:
    `Cross-entropy Loss <https://en.wikipedia.org/wiki/Cross-entropy>`_,
    Wikipedia

    `Multinomial Logistic Regression
    <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_, Wikipedia

  .. seealso:: This function is similar to
    :func:`optax.losses.softmax_cross_entropy`, but accepts integer labels
    instead of one-hot labels.

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(logits, jnp.floating)
  utils.check_subdtype(labels, jnp.integer)
  if where is not None and where.ndim != logits.ndim:
    where = jnp.expand_dims(where, axis)
  if isinstance(axis, int):
    axis = canonicalize_axis(axis, logits.ndim)
  elif isinstance(axis, tuple):
    # Move all "feature" dimensions to the end preserving axis ordering and
    # subsequent flattening "feature" dimensions to a single one.
    logit_axis = canonicalize_axes(axis, logits.ndim)
    batch_axis = tuple(x for x in range(logits.ndim) if x not in logit_axis)
    axis = len(batch_axis)
    logits = logits.transpose(batch_axis + logit_axis)
    logits = logits.reshape(logits.shape[:len(batch_axis)] + (-1,))
    if where is not None:
      where = where.transpose(batch_axis + logit_axis)
      where = where.reshape(where.shape[:len(batch_axis)] + (-1,))
  else:
    raise ValueError('Keyword argument \'axis\' must be of type \'int\' or '
                     f'\'tuple[int, ...]\' but actual type is {type(axis)}.')
  # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
  # we avoid subtracting the normalizer from all values, just from the values
  # for the correct labels.
  label_logits = jnp.take_along_axis(
      logits, jnp.expand_dims(labels, axis), axis=axis
  ).take(0, axis=axis)
  log_normalizers = jax.nn.logsumexp(logits, axis=axis, where=where)
  out = log_normalizers - label_logits
  if where is not None:
    out = jnp.where(jnp.any(where, axis), out, 0.0)
  return out


@functools.partial(
    warn_deprecated_function,
    replacement='softmax_cross_entropy_with_integer_labels',
)
def multiclass_logistic_loss(logits, labels):
  return softmax_cross_entropy_with_integer_labels(logits, labels)


_dot_last_dim = jnp.vectorize(jnp.dot, signature='(n),(n)->()')


def multiclass_hinge_loss(
    scores: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
) -> jax.Array:
  """Multiclass hinge loss.

  Args:
    scores: scores produced by the model (floats).
    labels: ground-truth integer labels.

  Returns:
    loss values

  References:
    `Hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`_, Wikipedia

  .. versionadded:: 0.2.3
  """
  one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
  return jnp.max(scores + 1.0 - one_hot_labels, axis=-1) - _dot_last_dim(
      scores, one_hot_labels
  )


def multiclass_perceptron_loss(
    scores: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
) -> jax.Array:
  """Multiclass perceptron loss.

  Args:
    scores: scores produced by the model.
    labels: ground-truth integer labels.

  Returns:
    loss values.

  References:
    Michael Collins. Discriminative training methods for Hidden Markov Models:
    Theory and experiments with perceptron algorithms. EMNLP 2002

  .. versionadded:: 0.2.2
  """
  one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
  return jnp.max(scores, axis=-1) - _dot_last_dim(scores, one_hot_labels)


def poly_loss_cross_entropy(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    *,
    epsilon: jax.typing.ArrayLike = 2.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  r"""Computes PolyLoss between logits and labels.

  The PolyLoss is a loss function that decomposes commonly
  used classification loss functions into a series of weighted
  polynomial bases. It is inspired by the Taylor expansion of
  cross-entropy loss and focal loss in the bases of :math:`(1 - P_t)^j`.

  .. math::
    L_{Poly} = \sum_1^\infty \alpha_j \cdot (1 - P_t)^j \\
    L_{Poly-N} = (\epsilon_1 + 1) \cdot (1 - P_t) + \ldots + \\
    (\epsilon_N + \frac{1}{N}) \cdot (1 - P_t)^N +
    \frac{1}{N + 1} \cdot (1 - P_t)^{N + 1} + \ldots = \\
    - \log(P_t) + \sum_{j = 1}^N \epsilon_j \cdot (1 - P_t)^j

  This function provides a simplified version of :math:`L_{Poly-N}`
  with only the coefficient of the first polynomial term being changed.

  Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Valid probability distributions (non-negative, sum to 1), e.g. a
      one hot encoding specifying the correct class for each input;
      must have a shape broadcastable to `[..., num_classes]`.
    epsilon: The coefficient of the first polynomial term.
      According to the paper, the following values are recommended:
      - For the ImageNet 2d image classification, epsilon = 2.0.
      - For the 2d Instance Segmentation and object detection, epsilon = -1.0.
      - It is also recommended to adjust this value based on the task, e.g. by
      using grid search.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    Poly loss between each prediction and the corresponding target
    distributions, with shape `[...]`.

  References:
    Leng et al, `PolyLoss: A Polynomial Expansion Perspective of Classification
    Loss Functions <https://arxiv.org/pdf/2204.12511.pdf>`_, 2022

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(logits, jnp.floating)
  utils.check_subdtype(labels, jnp.floating)
  p = jax.nn.softmax(logits, axis=axis, where=where)
  one_minus_pt = jnp.sum(labels * (1 - p), axis=axis, where=where)
  cross_entropy = softmax_cross_entropy(
      logits=logits, labels=labels, axis=axis, where=where
  )
  return cross_entropy + epsilon * one_minus_pt


def kl_divergence(
    log_predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].

  References:
    Kullback and Leibler, `On Information and Sufficiency
    <https://www.jstor.org/stable/2236703>`_, 1951

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(log_predictions, jnp.floating)
  utils.check_subdtype(targets, jnp.floating)
  loss = targets * (
      jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
  )
  return jnp.sum(loss, axis=axis, where=where)


def kl_divergence_with_log_targets(
    log_predictions: jax.typing.ArrayLike,
    log_targets: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Version of kl_div_loss where targets are given in log-space.

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    log_targets: Probabilities of target distribution with shape [..., dim].
      Expected to be in the log-space.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(log_predictions, jnp.floating)
  utils.check_subdtype(log_targets, jnp.floating)
  loss = jnp.exp(log_targets) * (log_targets - log_predictions)
  return jnp.sum(loss, axis=axis, where=where)


def generalized_kl_divergence(
    log_predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  """Computes the generalized Kullback-Leibler divergence loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  This function generalizes the standard Kullback-Leibler divergence to
  unnormalized probability distributions. Technically, this is the Bregman
  divergence generated by the convex function f(x) = x log x - x.

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    Generalized Kullback-Leibler divergence of predicted distribution from
    target distribution with shape [...].

  References:
    Boyd and Vandenberghe, `Convex Optimization
    <https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_,
    p. 90.

    L. M. Bregman, `The relaxation method of finding the common point of convex
    sets and its application to the solution of problems in convex programming
    <https://www.sciencedirect.com/science/article/pii/0041555367900407>`_,
    1967.

  .. versionadded:: 0.2.4
  """
  x = kl_divergence(log_predictions, targets, axis=axis, where=where)
  y = jnp.sum(jnp.exp(log_predictions) - targets, axis=axis, where=where)
  return x + y


@functools.partial(
    warn_deprecated_function, replacement='generalized_kl_divergence'
)
def convex_kl_divergence(
    log_predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  return generalized_kl_divergence(
      log_predictions, targets, axis=axis, where=where
  )


def ctc_loss_with_forward_probs(
    logits: jax.typing.ArrayLike,
    logit_paddings: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    label_paddings: jax.typing.ArrayLike,
    *,
    blank_id: int = 0,
    log_epsilon: jax.typing.ArrayLike = -1e5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  r"""Computes CTC loss and CTC forward-probabilities.

  The CTC loss is a loss function based on log-likelihoods of the model that
  introduces a special blank symbol :math:`\phi` to represent variable-length
  output sequences.

  Forward probabilities returned by this function, as auxiliary results, are
  grouped into two part: blank alpha-probability and non-blank alpha
  probability. Those are defined as follows:

  .. math::
    \alpha_{\mathrm{BLANK}}(t, n) =
    \sum_{\pi_{1:t-1}} p(\pi_t = \phi | \pi_{1:t-1}, y_{1:n-1}, \cdots), \\
    \alpha_{\mathrm{LABEL}}(t, n) =
    \sum_{\pi_{1:t-1}} p(\pi_t = y_n | \pi_{1:t-1}, y_{1:n-1}, \cdots).

  Here, :math:`\pi` denotes the alignment sequence in the reference
  [Graves et al, 2006] that is blank-inserted representations of ``labels``.
  The return values are the logarithms of the above probabilities.

  Args:
    logits: (B, T, K)-array containing logits of each class where B denotes
      the batch size, T denotes the max time frames in ``logits``, and K
      denotes the number of classes including a class for blanks.
    logit_paddings: (B, T)-array. Padding indicators for ``logits``. Each
      element must be either 1.0 or 0.0, and ``logitpaddings[b, t] == 1.0``
      denotes that ``logits[b, t, :]`` are padded values.
    labels: (B, N)-array containing reference integer labels where N denotes
      the max time frames in the label sequence.
    label_paddings: (B, N)-array. Padding indicators for ``labels``. Each
      element must be either 1.0 or 0.0, and ``labelpaddings[b, n] == 1.0``
      denotes that ``labels[b, n]`` is a padded label. In the current
      implementation, ``labels`` must be right-padded, i.e. each row
      ``labelpaddings[b, :]`` must be repetition of zeroes, followed by
      repetition of ones.
    blank_id: Id for blank token. ``logits[b, :, blank_id]`` are used as
      probabilities of blank symbols.
    log_epsilon: Numerically-stable approximation of log(+0).

  Returns:
    A tuple ``(loss_value, logalpha_blank, logalpha_nonblank)``. Here,
    ``loss_value`` is a (B,)-array containing the loss values for each sequence
    in the batch, ``logalpha_blank`` and ``logalpha_nonblank`` are
    (T, B, N+1)-arrays where the (t, b, n)-th element denotes
    \log \alpha_B(t, n) and \log \alpha_L(t, n), respectively, for ``b``-th
    sequence in the batch.

  References:
    Graves et al, `Connectionist temporal classification: labelling unsegmented
    sequence data with recurrent neural networks
    <https://dl.acm.org/doi/abs/10.1145/1143844.1143891>`_, 2006
  """

  utils.check_rank(logits, 3)
  utils.check_rank(labels, 2)
  utils.check_shapes_equal(labels, label_paddings)
  utils.check_shapes_equal(logits[..., 0], logit_paddings)
  batchsize, unused_maxinputlen, num_classes = logits.shape
  batchsize_of_labels, maxlabellen = labels.shape
  if batchsize_of_labels != batchsize:
    raise ValueError(
        f'Expected `labels` to have batch size {batchsize}, got'
        f' {batchsize_of_labels}.'
    )

  logprobs = jax.nn.log_softmax(logits)
  labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)

  # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
  repeat = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)
  repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

  logprobs_phi = logprobs[:, :, blank_id : blank_id + 1]  # [B, T, 1]
  logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

  one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
  logprobs_emit = jnp.einsum('btk,bnk->btn', logprobs, one_hot)
  logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

  logalpha_phi_init = (
      jnp.ones((batchsize, maxlabellen + 1)) * log_epsilon
  )  # [B, N]
  logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
  logalpha_emit_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon

  def update_phi_score(phi, added_score):
    # Update `phi[:, 1:]`` with adding `added_score` in log space.
    return jnp.concatenate(
        [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1
    )

  def loop_body(prev, x):
    prev_phi, prev_emit = prev
    # emit-to-phi epsilon transition, except if the next label is repetition
    prev_phi_orig = prev_phi
    prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

    logprob_emit, logprob_phi, pad = x

    # phi-to-emit transition
    next_emit = jnp.logaddexp(
        prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
    )
    # self-loop transition
    next_phi = prev_phi + logprob_phi
    # emit-to-phi blank transition only when the next label is repetition
    next_phi = update_phi_score(
        next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
    )

    pad = pad.reshape((batchsize, 1))
    next_emit = pad * prev_emit + (1.0 - pad) * next_emit
    next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

    return (next_phi, next_emit), (next_phi, next_emit)

  xs = (logprobs_emit, logprobs_phi, logit_paddings.transpose((1, 0)))
  _, (logalpha_phi, logalpha_emit) = jax.lax.scan(
      loop_body, (logalpha_phi_init, logalpha_emit_init), xs
  )

  # last row needs to be updated with the last epsilon transition
  logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
  logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

  # extract per_seq_loss
  one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
  per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)  # pylint:disable=invalid-unary-operand-type

  return per_seq_loss, logalpha_phi, logalpha_emit


def ctc_loss(
    logits: jax.typing.ArrayLike,
    logit_paddings: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    label_paddings: jax.typing.ArrayLike,
    *,
    blank_id: int = 0,
    log_epsilon: jax.typing.ArrayLike = -1e5,
) -> jax.Array:
  """Computes CTC loss.

  See docstring for ``ctc_loss_with_forward_probs`` for details.

  Args:
    logits: (B, T, K)-array containing logits of each class where B denotes the
      batch size, T denotes the max time frames in ``logits``, and K denotes the
      number of classes including a class for blanks.
    logit_paddings: (B, T)-array. Padding indicators for ``logits``. Each
      element must be either 1.0 or 0.0, and ``logitpaddings[b, t] == 1.0``
      denotes that ``logits[b, t, :]`` are padded values.
    labels: (B, N)-array containing reference integer labels where N denotes the
      max time frames in the label sequence.
    label_paddings: (B, N)-array. Padding indicators for ``labels``. Each
      element must be either 1.0 or 0.0, and ``labelpaddings[b, n] == 1.0``
      denotes that ``labels[b, n]`` is a padded label. In the current
      implementation, ``labels`` must be right-padded, i.e. each row
      ``labelpaddings[b, :]`` must be repetition of zeroes, followed by
      repetition of ones.
    blank_id: Id for blank token. ``logits[b, :, blank_id]`` are used as
      probabilities of blank symbols.
    log_epsilon: Numerically-stable approximation of log(+0).

  Returns:
    (B,)-array containing loss values for each sequence in the batch.
  """
  per_seq_loss, _, _ = ctc_loss_with_forward_probs(
      logits,
      logit_paddings,
      labels,
      label_paddings,
      blank_id=blank_id,
      log_epsilon=log_epsilon,
  )
  return per_seq_loss


def sigmoid_focal_loss(
    logits: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
    *,
    alpha: Optional[jax.typing.ArrayLike] = None,
    gamma: jax.typing.ArrayLike = 2.0,
) -> jax.Array:
  r"""Sigmoid focal loss with numerical stability improvements.

  The focal loss is a dynamically scaled cross entropy loss, where the scaling
  factor decays to zero as confidence in the correct class increases. This
  addresses class imbalance by down-weighting easy examples and focusing on
  hard examples.

  This implementation uses log-space computation for the focal weight
  :math:`(1-p_t)^\gamma` to ensure numerical stability, especially for
  :math:`\gamma < 2` and extreme logit values.

  The loss is defined as:

  .. math::
    FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)

  where :math:`p_t` is the predicted probability of the correct class:

  .. math::
    p_t = \begin{cases}
      p & \text{if } y = 1 \\
      1-p & \text{if } y = 0
    \end{cases}

  and :math:`\alpha_t` is the weighting factor:

  .. math::
    \alpha_t = \begin{cases}
      \alpha & \text{if } y = 1 \\
      1-\alpha & \text{if } y = 0
    \end{cases}

  Args:
    logits: Array of unnormalized log probabilities, with shape `[..., ]`.
      The predictions for each example.
    labels: Array of labels with shape broadcastable to `logits`. Can be:
      - Binary labels `{0, 1}` for binary classification
      - Continuous labels `[0, 1]` for soft targets or label smoothing
    alpha: (optional) Weighting factor in range `(0, 1)` to balance positive vs
      negative examples. Default `None` (no weighting).
    gamma: Exponent of the modulating factor `(1 - p_t)`. Higher values focus
      more on hard examples. Default `2.0`.

  Returns:
    Focal loss values with shape identical to `logits`.

  References:
    Lin et al, `Focal Loss for Dense Object Detection
    <https://arxiv.org/abs/1708.02002>`_, 2017

  .. versionchanged:: 0.2.5
    Added numerical stability improvements using log-space computation.
    Added support for continuous labels in `[0, 1]`.
  """
  utils.check_subdtype(logits, jnp.floating)
  labels = jnp.astype(labels, logits.dtype)

  # Cross-entropy loss
  ce_loss = sigmoid_binary_cross_entropy(logits, labels)

  # Compute log(1-p_t) using logsumexp unconditionally
  log_p = jax.nn.log_sigmoid(logits)
  log_q = jax.nn.log_sigmoid(-logits)

  log_one_minus_p_t = jax.scipy.special.logsumexp(
      jnp.stack([log_p, log_q], axis=-1),
      axis=-1,
      b=jnp.stack([1 - labels, labels], axis=-1)
  )

  # Focal weight and final loss
  focal_weight = jnp.exp(gamma * log_one_minus_p_t)
  loss = ce_loss * focal_weight

  # Alpha weighting
  if alpha is None:
    return loss
  weighted = (alpha * labels + (1.0 - alpha) * (1.0 - labels)) * loss
  return weighted


def _multiclass_sparsemax_loss(
    scores: jax.typing.ArrayLike, label: jax.typing.ArrayLike
) -> jax.Array:
  scores = jnp.asarray(scores)
  proba = projections.projection_simplex(scores)
  # Fenchel conjugate of the Gini negentropy, defined by:
  # cumulant = jnp.dot(proba, scores) + 0.5 * jnp.dot(proba, (1 - proba)).
  scores = (scores - scores[label]).at[label].set(0.0)
  return jnp.dot(proba, jnp.where(proba, scores, 0.0)) + 0.5 * (
      1.0 - jnp.dot(proba, proba)
  )


def multiclass_sparsemax_loss(
    scores: jax.typing.ArrayLike,
    labels: jax.typing.ArrayLike,
) -> jax.Array:
  """Multiclass sparsemax loss.

  Args:
    scores: scores produced by the model.
    labels: ground-truth integer labels.

  Returns:
    loss values

  References:
    Martins et al, `From Softmax to Sparsemax: A Sparse Model of Attention and
    Multi-Label Classification <https://arxiv.org/abs/1602.02068>`, 2016.
  """
  return jax.vmap(_multiclass_sparsemax_loss)(scores, labels)
