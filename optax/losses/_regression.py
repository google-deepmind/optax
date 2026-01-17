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
"""Regression losses."""

from typing import Optional, Union

import jax
import jax.numpy as jnp
from optax._src import utils


def squared_error(
    predictions: jax.typing.ArrayLike,
    targets: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
  """Calculates the squared error for a set of predictions.

  Mean Squared Error can be computed as squared_error(a, b).mean().

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    elementwise squared differences, with same shape as `predictions`.

  .. note::
    l2_loss = 0.5 * squared_error, where the 0.5 term is standard in
    "Pattern Recognition and Machine Learning" by Bishop, but not
    "The Elements of Statistical Learning" by Tibshirani.
  """
  utils.check_subdtype(predictions, jnp.floating)
  if targets is not None:
    # Avoid broadcasting logic for "-" operator.
    utils.check_shapes_equal(predictions, targets)
  errors = predictions - targets if targets is not None else predictions
  return errors**2


def l2_loss(
    predictions: jax.typing.ArrayLike,
    targets: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
  """Calculates the L2 loss for a set of predictions.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    elementwise squared differences, with same shape as `predictions`.

  .. note::
    the 0.5 term is standard in "Pattern Recognition and Machine Learning"
    by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.
  """
  predictions = jnp.asarray(predictions)
  return 0.5 * squared_error(predictions, targets)


def huber_loss(
    predictions: jax.typing.ArrayLike,
    targets: Optional[jax.typing.ArrayLike] = None,
    *,
    delta: jax.typing.ArrayLike = 1.0,
) -> jax.Array:
  """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.
    delta: the bounds for the huber loss transformation, defaults at 1.

  Returns:
    elementwise huber losses, with the same shape of `predictions`.

  References:
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_, Wikipedia.
  """
  utils.check_subdtype(predictions, jnp.floating)
  errors = (predictions - targets) if (targets is not None) else predictions
  # 0.5 * err^2                  if |err| <= d
  # 0.5 * d^2 + d * (|err| - d)  if |err| > d
  abs_errors = jnp.abs(errors)
  quadratic = jnp.minimum(abs_errors, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_errors - quadratic
  return 0.5 * quadratic**2 + delta * linear


def log_cosh(
    predictions: jax.typing.ArrayLike,
    targets: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
  """Calculates the log-cosh loss for a set of predictions.

  log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
  for large x.  It is a twice differentiable alternative to the Huber loss.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    the log-cosh loss, with same shape as `predictions`.

  References:
    Chen et al, `Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder
    <https://openreview.net/pdf?id=rkglvsC9Ym>`, 2019
  """
  utils.check_subdtype(predictions, jnp.floating)
  errors = (predictions - targets) if (targets is not None) else predictions
  # log(cosh(x)) = log((exp(x) + exp(-x))/2) = log(exp(x) + exp(-x)) - log(2)
  return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)


def cosine_similarity(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    epsilon: jax.typing.ArrayLike = 0.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  r"""Computes the cosine similarity between targets and predictions.

  The cosine **similarity** is a measure of similarity between vectors defined
  as the cosine of the angle between them, which is also the inner product of
  those vectors normalized to have unit norm.

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    cosine similarity measures, with shape `[...]`.

  References:
    `Cosine similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_,
    Wikipedia.

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(predictions, jnp.floating)
  utils.check_subdtype(targets, jnp.floating)
  a = predictions
  b = targets

  # dot = (a * b).sum(axis=axis, where=where)
  # a_norm2 = jnp.square(a).sum(axis=axis, where=where)
  # b_norm2 = jnp.square(b).sum(axis=axis, where=where)
  # return dot / jnp.sqrt((a_norm2 * b_norm2))

  a_norm2 = jnp.square(a).sum(axis=axis, where=where, keepdims=True)
  b_norm2 = jnp.square(b).sum(axis=axis, where=where, keepdims=True)
  a_norm = jnp.sqrt(a_norm2.clip(epsilon))
  b_norm = jnp.sqrt(b_norm2.clip(epsilon))
  a_unit = a / a_norm
  b_unit = b / b_norm
  return (a_unit * b_unit).sum(axis=axis, where=where)


def cosine_distance(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    epsilon: jax.typing.ArrayLike = 0.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> jax.Array:
  r"""Computes the cosine distance between targets and predictions.

  The cosine **distance**, implemented here, measures the **dissimilarity**
  of two vectors as the opposite of cosine **similarity**: `1 - cos(\theta)`.

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    cosine distances, with shape `[...]`.

  References:
    `Cosine distance
    <https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance>`_,
    Wikipedia.

  .. versionchanged:: 0.2.4
     Added ``axis`` and ``where`` arguments.
  """
  utils.check_subdtype(predictions, jnp.floating)
  utils.check_subdtype(targets, jnp.floating)
  # cosine distance = 1 - cosine similarity.
  return 1.0 - cosine_similarity(
      predictions, targets, epsilon=epsilon, axis=axis, where=where
  )


def poisson_nll_loss(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    *,
    log_input: bool = True,
    full: bool = False,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """Computes the Poisson negative log-likelihood loss.

  This loss is useful for regression problems where the targets represent
  counts (non-negative integers), following a Poisson distribution.

  Args:
    predictions: Predicted values. If ``log_input=True``, these are in
      log-space. Shape can be any broadcastable shape with ``targets``.
    targets: Ground truth count values (non-negative). Shape can be any
      broadcastable shape with ``predictions``.
    log_input: If ``True`` (default), ``predictions`` are assumed to be in
      log-space and will be exponentiated. If ``False``, ``predictions`` are
      treated as-is.
    full: If ``True``, compute the full loss including the Stirling
      approximation of the log-factorial term: ``log(target!)``. If ``False``
      (default), this constant term is omitted as it doesn't affect
      optimization.
    eps: Small constant for numerical stability when computing logarithms.

  Returns:
    The Poisson NLL loss, element-wise with shape matching the broadcasted
    shape of ``predictions`` and ``targets``.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> jnp.set_printoptions(precision=4)
    >>> # Predictions in log-space (default)
    >>> log_predictions = jnp.array([0.5, 1.0, 1.5])
    >>> targets = jnp.array([1.0, 2.0, 3.0])
    >>> loss = optax.poisson_nll_loss(log_predictions, targets)
    >>> print(loss)
    [1.1487 1.0986 1.4306]

    >>> # Predictions in natural space
    >>> predictions = jnp.exp(log_predictions)
    >>> loss_natural = optax.poisson_nll_loss(
    ...     predictions, targets, log_input=False)
    >>> print(loss_natural)
    [1.1487 1.0986 1.4306]

    >>> # With full loss (including factorial term)
    >>> loss_full = optax.poisson_nll_loss(
    ...     log_predictions, targets, full=True)
    >>> print(loss_full)
    [1.1487 1.7918 3.0078]

  References:
    `Poisson Distribution
    <https://en.wikipedia.org/wiki/Poisson_distribution>`_, Wikipedia

  .. versionadded:: 0.2.5
  """
  utils.check_subdtype(predictions, jnp.floating)
  targets = jnp.asarray(targets, dtype=predictions.dtype)

  if log_input:
    loss = jnp.exp(predictions) - targets * predictions
  else:
    predictions = jnp.clip(predictions, eps, None)
    loss = predictions - targets * jnp.log(predictions + eps)

  if full:
    stirling_approx = (
        targets * jnp.log(targets + eps) - targets +
        0.5 * jnp.log(2.0 * jnp.pi * (targets + eps))
    )
    stirling_approx = jnp.where(targets > 1, stirling_approx, 0.0)
    loss = loss + stirling_approx

  return loss


def gaussian_nll_loss(
    predictions: jax.typing.ArrayLike,
    targets: jax.typing.ArrayLike,
    var: jax.typing.ArrayLike,
    *,
    full: bool = False,
    eps: jax.typing.ArrayLike = 1e-6,
) -> jax.Array:
  """Computes the Gaussian negative log-likelihood loss.

  This loss is useful for regression problems where the model predicts both
  a mean and a variance (or standard deviation), enabling the model to express
  uncertainty in its predictions.

  Args:
    predictions: Predicted mean values. Shape can be any shape broadcastable
      with ``targets`` and ``var``.
    targets: Ground truth values. Shape can be any shape broadcastable with
      ``predictions`` and ``var``.
    var: Predicted variance values (must be non-negative). Can be a scalar,
      per-sample, or per-element variance. Shape must be broadcastable with
      ``predictions`` and ``targets``.
    full: If ``True``, include the constant term ``0.5 * log(2π)``. If
      ``False`` (default), omit this term as it doesn't affect optimization.
    eps: Small constant for numerical stability, used as minimum variance.

  Returns:
    The Gaussian NLL loss, element-wise with shape matching the broadcasted
    shape of ``predictions``, ``targets``, and ``var``.

  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> jnp.set_printoptions(precision=4)
    >>> predictions = jnp.array([1.0, 2.0, 3.0])
    >>> targets = jnp.array([1.2, 1.8, 3.1])
    >>> var = jnp.array([0.5, 0.5, 0.5])
    >>> loss = optax.gaussian_nll_loss(predictions, targets, var)
    >>> print(loss)
    [0.3866 0.4266 0.3566]

    >>> # With scalar variance
    >>> loss_scalar_var = optax.gaussian_nll_loss(
    ...     predictions, targets, var=1.0)
    >>> print(loss_scalar_var)
    [0.02 0.02 0.005]

    >>> # With full loss (including constant)
    >>> loss_full = optax.gaussian_nll_loss(
    ...     predictions, targets, var, full=True)
    >>> print(loss_full)
    [1.3059 1.3459 1.2759]

  References:
    `Normal Distribution
    <https://en.wikipedia.org/wiki/Normal_distribution>`_, Wikipedia

  .. versionadded:: 0.2.5
  """
  utils.check_subdtype(predictions, jnp.floating)
  targets = jnp.asarray(targets, dtype=predictions.dtype)
  var = jnp.asarray(var, dtype=predictions.dtype)

  var = jnp.maximum(var, eps)

  loss = 0.5 * (jnp.log(var) + (predictions - targets) ** 2 / var)

  if full:
    loss = loss + 0.5 * jnp.log(2.0 * jnp.pi)

  return loss
