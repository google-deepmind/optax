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
"""Tests for optax.examples.lookahead_mnist."""
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing.absltest import mock
import numpy as np
from optax.examples import datasets
from optax.examples import lookahead_mnist
import tensorflow as tf


class LookaheadMnistTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    self.logits = np.log(
        np.array([[.4, .3, .3], [.5, .4, .1], [.2, .1, .7], [.7, .2, .1]]))
    self.correct_loss = -0.5 * (np.log(0.4) + np.log(0.7))
    self.correct_accuracy = 0.75
    batch_size = 2
    # We use the identity function as model in the tests below so that we can
    # obtain the desired model output (the logits above) by passing them as
    # input "image" to the model.
    data = {'image': self.logits, 'label': self.labels}
    self.dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

  def test_categorical_crossentropy(self):
    """Tests the loss function."""
    loss = lookahead_mnist.categorical_crossentropy(self.logits, self.labels)
    self.assertAlmostEqual(loss, self.correct_loss, places=6)

  def test_accuracy(self):
    """Tests the accuracy calculation."""
    accuracy = lookahead_mnist.accuracy(self.logits, self.labels)
    self.assertEqual(accuracy, self.correct_accuracy)

  def test_model_accuracy(self):
    """Tests the calculation of metrics on a dataset."""
    accuracy = lookahead_mnist.model_accuracy(lambda x: x,
                                              self.dataset.as_numpy_iterator())
    self.assertEqual(accuracy, self.correct_accuracy)

  @flagsaver.flagsaver(hidden_dims=[3, 5])
  def test_main(self):
    """Checks that the example runs successfully."""
    with mock.patch.object(
        datasets, 'load_image_dataset', return_value=self.dataset):
      lookahead_mnist.main({})


if __name__ == '__main__':
  absltest.main()
