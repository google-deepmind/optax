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
from absl.testing.absltest import mock

import numpy as np
import tensorflow as tf

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
import lookahead_mnist  # Located in the examples folder.
# pylint: enable=g-bad-import-order


class LookaheadMnistTest(absltest.TestCase):

  def test_lookahead_example_can_fit_linear_mock_data(self):
    """Checks that the lookahead example can fit linear mock data."""
    lookahead_mnist.LEARNING_RATE = 1e-2
    lookahead_mnist.HIDDEN_SIZES = (1,)
    data = {
        'image': np.arange(-0.5, 0.5, 0.1).reshape(10, 1, 1),
        'label': np.array([[1, 0]] * 4 + [[0, 1]] * 6)
    }
    dataset = tf.data.Dataset.from_tensor_slices(data).repeat(8).batch(10)
    with mock.patch.object(
        datasets, 'load_image_dataset', return_value=dataset):
      final_accuracy = lookahead_mnist.main({})

    self.assertEqual(final_accuracy, 1.)


if __name__ == '__main__':
  absltest.main()
