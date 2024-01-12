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

from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp


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

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: Each element is the unnormalized log probability of a binary
      prediction. See note about compatibility with `labels` above.
    labels: Binary labels whose values are {0,1} or multi-class target
      probabilities. See note about compatibility with `logits` above.

  Returns:
    cross entropy for each binary prediction, same shape as `logits`.
  """
  chex.assert_type([logits], float)
  labels = labels.astype(logits.dtype)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


def hinge_loss(
    predictor_outputs: chex.Array,
    targets: chex.Array
) -> chex.Array:
  """Computes the hinge loss for binary classification.

  Args:
    predictor_outputs: Outputs of the decision function.
    targets: Target values. Target values should be strictly in the set {-1, 1}.

  Returns:
    Binary Hinge Loss.
  """
  return jnp.maximum(0, 1 - predictor_outputs * targets)


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
      must have a shape broadcastable to `[..., num_classes]`.

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


def poly_loss_cross_entropy(
    logits: chex.Array, labels: chex.Array, epsilon: float = 2.0
) -> chex.Array:
  r"""Computes PolyLoss between logits and labels.

  The PolyLoss is a loss function that decomposes commonly
  used classification loss functions into a series of weighted
  polynomial bases. It is inspired by the Taylor expansion of
  cross-entropy loss and focal loss in the bases of :math:`(1 − P_t)^j`.

  .. math::
    L_{Poly} = \sum_1^\infty \alpha_j \cdot (1 - P_t)^j \\
    L_{Poly-N} = (\epsilon_1 + 1) \cdot (1 - P_t) + \ldots + \\
    (\epsilon_N + \frac{1}{N}) \cdot (1 - P_t)^N +
    \frac{1}{N + 1} \cdot (1 - P_t)^{N + 1} + \ldots = \\
    - \log(P_t) + \sum_{j = 1}^N \epsilon_j \cdot (1 - P_t)^j

  This function provides a simplified version of :math:`L_{Poly-N}`
  with only the coefficient of the first polynomial term being changed.

  References:
    [Zhaoqi Leng et al, 2022](https://arxiv.org/pdf/2204.12511.pdf)

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

  Returns:
    Poly loss between each prediction and the corresponding target
    distributions, with shape `[...]`.
  """
  chex.assert_type([logits, labels], float)
  one_minus_pt = jnp.sum(labels * (1 - jax.nn.softmax(logits)), axis=-1)
  cross_entropy = softmax_cross_entropy(logits=logits, labels=labels)
  return cross_entropy + epsilon * one_minus_pt


def kl_divergence(
    log_predictions: chex.Array,
    targets: chex.Array
) -> chex.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  References:
    [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  chex.assert_type([log_predictions, targets], float)
  loss = targets * (
      jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
  )
  return jnp.sum(loss, axis=-1)


def kl_divergence_with_log_targets(
    log_predictions: chex.Array,
    log_targets: chex.Array
) -> chex.Array:
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


def convex_kl_divergence(
    log_predictions: chex.Array, targets: chex.Array
) -> chex.Array:
  """Computes a convex version of the Kullback-Leibler divergence loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.
  This version is jointly convex in p (targets) and q (log_predictions).

  References:
    [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  return kl_divergence(log_predictions, targets) + jnp.sum(
      jnp.exp(log_predictions) - targets, axis=-1
  )


def ctc_loss_with_forward_probs(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5
) -> Tuple[chex.Array, chex.Array, chex.Array]:
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
  per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)  # pylint:disable=invalid-unary-operand-type

  return per_seq_loss, logalpha_phi, logalpha_emit


def ctc_loss(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5
) -> chex.Array:
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


def sigmoid_focal_loss(
    logits: chex.Array,
    labels: chex.Array,
    alpha: Optional[float] = None,
    gamma: float = 2.,
) ->  chex.Array:
  """Sigmoid focal loss.

  The focal loss is a re-weighted cross entropy for unbalanced problems.
  Use this loss function if classes are not mutually exclusive.
  See `sigmoid_binary_cross_entropy` for more information.

  References:
    Lin et al. 2018. https://arxiv.org/pdf/1708.02002.pdf

  Args:
    logits: Array of floats. The predictions for each example.
      The predictions for each example.
    labels: Array of floats. Labels and logits must have
      the same shape. The label array must contain the binary
      classification labels for each element in the data set
      (0 for the out-of-class and 1 for in-class).
    alpha: (optional) Weighting factor in range (0,1) to balance
      positive vs negative examples. Default None (no weighting).
    gamma: Exponent of the modulating factor (1 - p_t).
      Balances easy vs hard examples.

  Returns:
    A loss value array with a shape identical to the logits and target
    arrays.
  """
  alpha = -1 if alpha is None else alpha

  chex.assert_type([logits], float)
  labels = labels.astype(logits.dtype)
  # see also the original paper's implementation at:
  # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
  p = jax.nn.sigmoid(logits)
  ce_loss = sigmoid_binary_cross_entropy(logits, labels)
  p_t = p * labels + (1 - p) * (1 - labels)
  loss = ce_loss * ((1 - p_t) ** gamma)

  weighted = lambda loss_arg: (alpha * labels
                               + (1 - alpha) * (1 - labels)
                              )*loss_arg
  not_weighted = lambda loss_arg: loss_arg

  loss = jax.lax.cond(alpha >= 0,
                      weighted,
                      not_weighted,
                      loss)

  return loss
