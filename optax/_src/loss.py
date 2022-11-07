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
"""Standard losses used in optimisation.

We provide implementations of the most canonical losses used in deep
learning. These operate transparently on batches, and do not perform any
reduction over the batch dimensions, leaving it to the user to, for instance,
mean or sum losses across batch dimensions.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from optax._src import utils


def l2_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the L2 loss for a set of predictions.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  References:
    [Chris Bishop, 2006](https://bit.ly/3eeP0ga)

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`;
      if not provided then it is assumed to be a vector of zeros.

  Returns:
    elementwise squared differences, with same shape as `predictions`.
  """
  chex.assert_type([predictions], float)
  if targets is not None:
    # Avoid broadcasting logic for "-" operator.
    chex.assert_equal_shape((predictions, targets))
  errors = (predictions - targets) if (targets is not None) else predictions
  return 0.5 * (errors)**2


def huber_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
    delta: float = 1.) -> chex.Array:
  """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  References:
    [Huber, 1964](www.projecteuclid.org/download/pdf_1/euclid.aoms/1177703732)

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`;
      if not provided then it is assumed to be a vector of zeros.
    delta: the bounds for the huber loss transformation, defaults at 1.

  Returns:
    elementwise huber losses, with the same shape of `predictions`.
  """
  chex.assert_type([predictions], float)
  errors = (predictions - targets) if (targets is not None) else predictions
  # 0.5 * err^2                  if |err| <= d
  # 0.5 * d^2 + d * (|err| - d)  if |err| > d
  abs_errors = jnp.abs(errors)
  quadratic = jnp.minimum(abs_errors, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_errors - quadratic
  return 0.5 * quadratic ** 2 + delta * linear


def smooth_labels(
    labels: chex.Array,
    alpha: float,
) -> jnp.ndarray:
  """Apply label smoothing.

  Label smoothing is often used in combination with a cross-entropy loss.
  Smoothed labels favour small logit gaps, and it has been shown that this can
  provide better model calibration by preventing overconfident predictions.

  References:
    [Müller et al, 2019](https://arxiv.org/pdf/1906.02629.pdf)

  Args:
    labels: one hot labels to be smoothed.
    alpha: the smoothing factor, the greedy category with be assigned
      probability `(1-alpha) + alpha / num_categories`

  Returns:
    a smoothed version of the one hot input labels.

  """
  chex.assert_type([labels], float)
  num_categories = labels.shape[-1]
  return (1.0 - alpha) * labels + alpha / num_categories


def sigmoid_binary_cross_entropy(logits, labels):
  """Computes element-wise sigmoid cross entropy given logits and labels.

  This can be used to measure the error in discrete classification tasks in
  which each class is an independent binary prediction and different classes
  are not mutually exclusive. This may be used for multilabel image
  classification for instance a model may predict that an image contains both a
  cat and a dog.

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: Each element is the unnormalized log probability of a binary
      prediction.
    labels: The target probabilities, must have a shape broadcastable to that of
      `logits`.

  Returns:
    cross entropy for each binary prediction, same shape as `logits`.
  """
  chex.assert_type([logits], float)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  Measures the probability error in discrete classification tasks in which
  the classes are mutually exclusive (each entry is in exactly one class).
  For example, each CIFAR-10 image is labeled with one and only one label:
  an image can be a dog or a truck, but not both.

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Valid probability distributions (non-negative, sum to 1), e.g a
      one hot encoding specifying the correct class for each input;
      must have a shape broadcastable to `[..., num_classes]``

  Returns:
    cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
  """
  chex.assert_type([logits], float)
  return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def softmax_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
  """Computes softmax cross entropy between sets of logits and integer labels.

  Measures the probability error in discrete classification tasks in which
  the classes are mutually exclusive (each entry is in exactly one class).
  For example, each CIFAR-10 image is labeled with one and only one label:
  an image can be a dog or a truck, but not both.

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
      `[...]`.

  Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
  """
  chex.assert_type([logits], float)
  chex.assert_type([labels], int)
  # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
  # we avoid subtracting the normalizer from all values, just from the values
  # for the correct labels.
  logits_max = jnp.max(logits, axis=-1, keepdims=True)
  logits -= jax.lax.stop_gradient(logits_max)
  label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
  log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
  return log_normalizers - label_logits


def cosine_similarity(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.,
) -> chex.Array:
  r"""Computes the cosine similarity between targets and predictions.

  The cosine **similarity** is a measure of similarity between vectors defined
  as the cosine of the angle between them, which is also the inner product of
  those vectors normalized to have unit norm.

  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Cosine_similarity)

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.

  Returns:
    cosine similarity measures, with shape `[...]`.
  """
  chex.assert_type([predictions, targets], float)
  # vectorize norm fn, to treat all dimensions except the last as batch dims.
  batched_norm_fn = jnp.vectorize(
      utils.safe_norm, signature='(k)->()', excluded={1})
  # normalise the last dimension of targets and predictions.
  unit_targets = targets / jnp.expand_dims(
      batched_norm_fn(targets, epsilon), axis=-1)
  unit_predictions = predictions / jnp.expand_dims(
      batched_norm_fn(predictions, epsilon), axis=-1)
  # return cosine similarity.
  return jnp.sum(unit_targets * unit_predictions, axis=-1)


def cosine_distance(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.,
) -> chex.Array:
  r"""Computes the cosine distance between targets and predictions.

  The cosine **distance**, implemented here, measures the **dissimilarity**
  of two vectors as the opposite of cosine **similarity**: `1 - cos(\theta)`.

  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Cosine_similarity)

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.

  Returns:
    cosine distances, with shape `[...]`.
  """
  chex.assert_type([predictions, targets], float)
  # cosine distance = 1 - cosine similarity.
  return 1. - cosine_similarity(predictions, targets, epsilon)


def log_cosh(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the log-cosh loss for a set of predictions.

  log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
  for large x.  It is a twice differentiable alternative to the Huber loss.

  References:
    [Chen et al, 2019](https://openreview.net/pdf?id=rkglvsC9Ym)

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`;
      if not provided then it is assumed to be a vector of zeros.

  Returns:
    the log-cosh loss, with same shape as `predictions`.
  """
  chex.assert_type([predictions], float)
  errors = (predictions - targets) if (targets is not None) else predictions
  # log(cosh(x)) = log((exp(x) + exp(-x))/2) = log(exp(x) + exp(-x)) - log(2)
  return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)


def ctc_loss_with_forward_probs(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5) -> Tuple[chex.Array, chex.Array, chex.Array]:
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

  References:
    [Graves et al, 2006](https://dl.acm.org/doi/abs/10.1145/1143844.1143891)

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
  """

  chex.assert_rank(logits, 3)
  chex.assert_rank(labels, 2)
  batchsize, unused_maxinputlen, num_classes = logits.shape
  batchsize_of_labels, maxlabellen = labels.shape
  chex.assert_equal(batchsize, batchsize_of_labels)
  chex.assert_equal(labels.shape, label_paddings.shape)
  chex.assert_equal(logits.shape[:2], logit_paddings.shape)

  logprobs = jax.nn.log_softmax(logits)
  labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)

  # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
  repeat = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)
  repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

  logprobs_phi = logprobs[:, :, blank_id:blank_id + 1]  # [B, T, 1]
  logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

  one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
  logprobs_emit = jnp.einsum('btk,bnk->btn', logprobs, one_hot)
  logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

  logalpha_phi_init = jnp.ones(
      (batchsize, maxlabellen + 1)) * log_epsilon  # [B, N]
  logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
  logalpha_emit_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon

  def update_phi_score(phi, added_score):
    # Update `phi[:, 1:]`` with adding `added_score` in log space.
    return jnp.concatenate(
        [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1)

  def loop_body(prev, x):
    prev_phi, prev_emit = prev
    # emit-to-phi epsilon transition, except if the next label is repetition
    prev_phi_orig = prev_phi
    prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

    logprob_emit, logprob_phi, pad = x

    # phi-to-emit transition
    next_emit = jnp.logaddexp(prev_phi[:, :-1] + logprob_emit,
                              prev_emit + logprob_emit)
    # self-loop transition
    next_phi = prev_phi + logprob_phi
    # emit-to-phi blank transition only when the next label is repetition
    next_phi = update_phi_score(
        next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat))

    pad = pad.reshape((batchsize, 1))
    next_emit = pad * prev_emit + (1.0 - pad) * next_emit
    next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

    return (next_phi, next_emit), (next_phi, next_emit)

  xs = (logprobs_emit, logprobs_phi, logit_paddings.transpose((1, 0)))
  _, (logalpha_phi,
      logalpha_emit) = jax.lax.scan(loop_body,
                                    (logalpha_phi_init, logalpha_emit_init), xs)

  # last row needs to be updated with the last epsilon transition
  logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
  logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

  # extract per_seq_loss
  one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
  per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)

  return per_seq_loss, logalpha_phi, logalpha_emit


def ctc_loss(logits: chex.Array,
             logit_paddings: chex.Array,
             labels: chex.Array,
             label_paddings: chex.Array,
             blank_id: int = 0,
             log_epsilon: float = -1e5) -> chex.Array:
  """Computes CTC loss.

  See docstring for ``ctc_loss_with_forward_probs`` for details.

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
    (B,)-array containing loss values for each sequence in the batch.
  """
  per_seq_loss, _, _ = ctc_loss_with_forward_probs(
      logits, logit_paddings, labels, label_paddings,
      blank_id=blank_id, log_epsilon=log_epsilon)
  return per_seq_loss


def kl_divergence(log_predictions: chex.Array,
                  targets: chex.Array) -> chex.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  References:
    [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)

  Args:
    log_predictions: Probabilities of predicted distribution with shape
      [..., dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  chex.assert_type([log_predictions, targets], float)
  loss = targets * (jnp.log(targets) - log_predictions)
  return jnp.sum(loss, axis=-1)


def kl_divergence_with_log_targets(log_predictions: chex.Array,
                                   log_targets: chex.Array) -> chex.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Version of kl_div_loss where targets are given in log-space.

  Args:
    log_predictions: Probabilities of predicted distribution with shape
      [..., dim]. Expected to be in the log-space to avoid underflow.
    log_targets: Probabilities of target distribution with shape [..., dim].
      Expected to be in the log-space.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  chex.assert_type([log_predictions, log_targets], float)
  loss = jnp.exp(log_targets) * (log_targets - log_predictions)
  return jnp.sum(loss, axis=-1)


def hinge_loss(predictor_outputs: chex.Array,
               targets: chex.Array) -> chex.Array:
  """Computes the hinge loss for binary classification.

  Args:
    predictor_outputs: Outputs of the decision function.
    targets: Target values. Target values should be strictly in the set {-1, 1}.

  Returns:
    Binary Hinge Loss.
  """
  return jnp.maximum(0, 1 - predictor_outputs * targets)
