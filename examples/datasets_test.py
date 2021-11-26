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
"""Tests for optax.examples.datasets."""
from typing import Iterable, List

from absl.testing import absltest
from absl.testing.absltest import mock
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
# pylint: enable=g-bad-import-order


def _batch_array(array: np.ndarray, batch_size: int) -> List[np.ndarray]:
  """Splits an array into batches."""
  split_indices = np.arange(batch_size, array.shape[0], batch_size)
  return np.split(array, split_indices)


def _assert_batches_almost_equal(actual: Iterable[np.ndarray],
                                 desired: Iterable[np.ndarray]) -> None:
  """Asserts that two dataset iterables are almost equal per batch."""
  for batch, correct_batch in zip(actual, desired):
    np.testing.assert_almost_equal(batch, correct_batch)


class DatasetsTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self._images = np.array([[50], [150], [250]])
    self._one_hot_labels = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    self._test_dataset = tf.data.Dataset.from_tensor_slices({
        'image': self._images,
        'label': np.argmax(self._one_hot_labels, axis=1),
    })
    self._batch_size = 2

  def test_load_image_dataset(self) -> None:
    """Checks datasets are loaded and preprocessed correctly."""
    with mock.patch.object(tfds, 'load', return_value=self._test_dataset):
      dataset = datasets.load_image_dataset(
          'unused_by_mock', self._batch_size, shuffle=False)

    with self.subTest('test_images'):
      image_batches = dataset.map(lambda x: x['image']).as_numpy_iterator()
      correct_image_batches = _batch_array(self._images / 255.,
                                           self._batch_size)
      _assert_batches_almost_equal(image_batches, correct_image_batches)

    with self.subTest('test_labels'):
      label_batches = dataset.map(lambda x: x['label']).as_numpy_iterator()
      correct_label_batches = _batch_array(self._one_hot_labels,
                                           self._batch_size)
      _assert_batches_almost_equal(label_batches, correct_label_batches)


if __name__ == '__main__':
  absltest.main()
