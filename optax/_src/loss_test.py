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
"""Tests for optax._src.loss."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import loss


class L2LossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([-2., -1., 0.5, 1.])
    self.ts = jnp.array([-1.5, 0., -1, 1.])
    # compute expected outputs in numpy.
    self.exp = 0.5 * (self.ts - self.ys) ** 2

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys[0], self.ts[0]), self.exp[0])

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys, self.ts), self.exp)


class HuberLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([-2.0, 0.5, 0., 0.5, 2.0, 4.0, 132.])
    self.ts = np.array([0.0, -0.5, 0., 1., 1.0, 2.0, 0.3])
    # computed expected outputs manually.
    self.exp = np.array([1.5, 0.5, 0., 0.125, 0.5, 1.5, 131.2])

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(loss.huber_loss)(self.ys[0], self.ts[0], delta=1.0),
        self.exp[0])

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(loss.huber_loss)(self.ys, self.ts, delta=1.0),
        self.exp)


class SmoothLabelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    # compute expected outputs in numpy.
    self.exp_alpha_zero = self.ts
    self.exp_alpha_zero_point_one = 0.9 * self.ts + 0.1 / self.ts.shape[-1]
    self.exp_alpha_one = jnp.ones_like(self.ts) / self.ts.shape[-1]

  @chex.all_variants()
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 0.),
        self.exp_alpha_zero[0], atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 0.1),
        self.exp_alpha_zero_point_one[0], atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 1.),
        self.exp_alpha_one[0], atol=1e-4)

  @chex.all_variants()
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 0.),
        self.exp_alpha_zero, atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 0.1),
        self.exp_alpha_zero_point_one, atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 1.),
        self.exp_alpha_one, atol=1e-4)


class SoftmaxCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
    self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    # taken expected outputs from rlax.
    self.exp = np.array([9.00013, 3.0696733], dtype=np.float32)

  @chex.all_variants()
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.softmax_cross_entropy)(self.ys[0], self.ts[0]),
        self.exp[0], atol=1e-4)

  @chex.all_variants()
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.softmax_cross_entropy)(self.ys, self.ts),
        self.exp, atol=1e-4)


class SigmoidCrossEntropyTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(preds=np.array([-1e+09, -1e-09]),
           labels=np.array([1., 0.]),
           expected=5e+08),
      dict(preds=np.array([-1e+09, -1e-09]),
           labels=np.array([0., 1.]),
           expected=0.3465736),
      dict(preds=np.array([1e+09, 1e-09]),
           labels=np.array([1., 0.]),
           expected=0.3465736),
      dict(preds=np.array([1e+09, 1e-09]),
           labels=np.array([0., 1.]),
           expected=5e+08),
      dict(preds=np.array([-1e+09, 1e-09]),
           labels=np.array([1., 0.]),
           expected=5e+08),
      dict(preds=np.array([-1e+09, 1e-09]),
           labels=np.array([0., 1.]),
           expected=0.3465736),
      dict(preds=np.array([1e+09, -1e-09]),
           labels=np.array([1., 0.]),
           expected=0.3465736),
      dict(preds=np.array([1e+09, -1e-09]),
           labels=np.array([0., 1.]),
           expected=5e+08),
      dict(preds=np.array([0., 0.]),
           labels=np.array([1., 0.]),
           expected=0.6931472),
      dict(preds=np.array([0., 0.]),
           labels=np.array([0., 1.]),
           expected=0.6931472),
  )
  def testSigmoidCrossEntropy(self, preds, labels, expected):
    tested = jnp.mean(loss.sigmoid_binary_cross_entropy(preds, labels))
    np.testing.assert_allclose(tested, expected, rtol=1e-6, atol=1e-6)


class CosineDistanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
    self.ts = np.array([[0., 1.2, 0.2], [1., -0.3, 0.]], dtype=np.float32)
    # distance computed expected output from `scipy 1.20`.
    self.exp = np.array([0.9358251989, 1.0464068465], dtype=np.float32)

  @chex.all_variants()
  def test_scalar_distance(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_distance)(self.ys[0], self.ts[0]),
        self.exp[0], atol=1e-4)

  @chex.all_variants()
  def test_scalar_similarity(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_similarity)(self.ys[0], self.ts[0]),
        1. - self.exp[0], atol=1e-4)

  @chex.all_variants()
  def test_batched_distance(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_distance)(self.ys, self.ts),
        self.exp, atol=1e-4)

  @chex.all_variants()
  def test_batched_similarity(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_similarity)(self.ys, self.ts),
        1. - self.exp, atol=1e-4)


# TODO(b/188419459): add test for grad and second order grad.
class LogCoshTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Test large values for overflow
    self.ys = jnp.array([500, -2., -1., 0.5, 1.])
    self.ts = jnp.array([-200, -1.5, 0., -1, 1.])
    # computed using tensorflow.keras.losses.log_cosh v2.4.1
    self.exp = jnp.array([699.3068, 0.12011445, 0.4337809, 0.85544014, 0.])
    self.exp_ys_only = jnp.array(
        [499.30685, 1.3250027, 0.4337809, 0.12011451, 0.43378082])

  @chex.all_variants
  def test_scalar(self):
    out = self.variant(loss.log_cosh)(self.ys[0], self.ts[0])
    np.testing.assert_allclose(out, self.exp[0], atol=1e-5)

  @chex.all_variants
  def test_batched(self):
    out = self.variant(loss.log_cosh)(self.ys, self.ts)
    np.testing.assert_allclose(out, self.exp, atol=1e-5)

  @chex.all_variants
  def test_scalar_predictions_only(self):
    out = self.variant(loss.log_cosh)(self.ys[0])
    np.testing.assert_allclose(out, self.exp_ys_only[0], atol=1e-5)

  @chex.all_variants
  def test_batched_predictions_only(self):
    out = self.variant(loss.log_cosh)(self.ys)
    np.testing.assert_allclose(out, self.exp_ys_only, atol=1e-5)


def _lengths_to_paddings(lengths: chex.Array, maxlength: int) -> chex.Array:
  indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
  lengths = jnp.expand_dims(lengths, axis=-1)
  elem_valid = indices < lengths
  return np.logical_not(elem_valid).astype(np.float32)


def _average_ctc_loss(logprobs: chex.Array, logprob_paddings: chex.Array,
                      labels: chex.Array,
                      label_paddings: chex.Array) -> chex.Array:
  return jnp.average(
      loss.ctc_loss(logprobs, logprob_paddings, labels,
                    label_paddings))

class CTCTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  @chex.all_variants
  def test_with_one_to_one_alignment(self):
    # when inputsteps and outputsteps are equal, no blank will be allowed.
    batchsize = 8
    steps = 50
    nclasses = 40
    logits = np.random.randn(batchsize, steps, nclasses)
    labels = np.random.uniform(
        1, nclasses, size=(batchsize, steps)).astype(np.int32)

    # This function only covers the cases without same-label repetition.
    # `test_repeat_with_one_to_one_alignment` below complements those cases.
    # So, redraw the samples for satisfying the non-repetition constraint.
    for n in range(labels.shape[0]):
      for t in range(1, labels.shape[1]):
        while labels[n, t] == labels[n, t - 1]:
          labels[n, t] = np.random.uniform(1, nclasses)

    per_seq_loss = self.variant(loss.ctc_loss)(
        logits, np.zeros(logits.shape[:2]), labels, np.zeros(labels.shape))

    logprobs = jax.nn.log_softmax(logits)
    for b in range(batchsize):
      p = 0.0
      for t in range(steps):
        p += logprobs[b, t, labels[b, t]]
      np.testing.assert_allclose(jnp.array(-p), per_seq_loss[b],
                                 rtol=1e-6)

  @chex.all_variants
  def test_with_one_to_one_alignment_and_paddings(self):
    batch_size = 5
    nclasses = 13
    steps = 7
    logits = np.random.normal(size=[batch_size, steps, nclasses])
    logprobs = jax.nn.log_softmax(logits)

    labels = []
    for n in range(batch_size):
      row = list(range(1, nclasses))
      np.random.shuffle(row)
      labels.append(row[:steps])
    labels = np.array(labels)

    lengths = np.random.randint(3, 6, size=(batch_size,))
    paddings = _lengths_to_paddings(lengths, steps)

    actual_loss = self.variant(loss.ctc_loss)(
        logits, paddings, labels, paddings)

    value_and_grad = self.variant(jax.value_and_grad(_average_ctc_loss))
    unused_avg_loss, actual_gradients = value_and_grad(
        logits, paddings, labels, paddings)

    for n in range(batch_size):
      expected_loss = -sum(logprobs[n, t, k]
                           for t, k in enumerate(labels[n, :lengths[n]]))
      np.testing.assert_allclose(expected_loss, actual_loss[n])

      expected_gradients = np.array(jax.nn.softmax(logits[n]))
      expected_gradients[lengths[n]:] = 0.0
      for t, k in enumerate(labels[n, :lengths[n]]):
        expected_gradients[t, k] -= 1.0
      expected_gradients /= batch_size
      np.testing.assert_allclose(expected_gradients, actual_gradients[n],
                                 rtol=1e-6)

  @chex.all_variants
  def test_repeat_with_one_to_one_alignment(self):
    # test if it can correctly handle the same-label repetition.
    nclasses = 5
    labels = np.array([
        [1, 2, 2, 3],
        [2, 3, 4, 4],
        [1, 1, 1, 1],
        [1, 1, 2, 3],
        [1, 1, 1, 2],
    ])
    expected_alignment = [  # expected minimal alignment
        [1, 2, 0, 2, 3],
        [2, 3, 4, 0, 4],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 2, 3],
        [1, 0, 1, 0, 1, 2],
    ]
    batch_size = len(labels)
    label_lens = np.array([4] * batch_size)
    label_steps = 6
    # Designed to have two padding elements on the right.
    labels = np.pad(labels, [(0, 0), (0, label_steps - labels.shape[1])])
    label_paddings = _lengths_to_paddings(label_lens, label_steps)

    logit_lengths = np.array([len(seq) for seq in expected_alignment])
    logit_steps = max(logit_lengths)
    logits = np.random.randn(batch_size, logit_steps, nclasses)
    logit_paddings = _lengths_to_paddings(logit_lengths, logit_steps)

    per_seq_loss = self.variant(loss.ctc_loss)(
        logits, logit_paddings, labels, label_paddings)

    logprobs = jax.nn.log_softmax(logits)
    for n in range(batch_size):
      expected_loss = -sum(
          logprobs[n, t, k]
          for t, k in enumerate(expected_alignment[n]))
      np.testing.assert_allclose(
          jnp.array(expected_loss), per_seq_loss[n], rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
