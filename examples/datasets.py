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
"""Datasets used in the examples."""
import functools
from typing import Dict, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Split = tfds.Split


def _preprocess_image_dataset(element: Mapping[str, tf.Tensor],
                              num_labels: int) -> Dict[str, tf.Tensor]:
  """Casts image to floats in the range [0,1] and one-hot encodes the label."""
  rescaled_image = tf.cast(element['image'], tf.float32) / 255.
  one_hot_label = tf.one_hot(
      tf.cast(element['label'], tf.int32), num_labels, on_value=1, off_value=0)
  return {'image': rescaled_image, 'label': one_hot_label}


def load_image_dataset(dataset_name: str,
                       batch_size: int,
                       split: Split = Split.TRAIN,
                       *,
                       shuffle: bool = True,
                       buffer_size: int = 10000,
                       cache: bool = True) -> tf.data.Dataset:
  """Loads an pre-processes an image dataset from tensorflow_datasets.

  The dataset is pre-processed so as to be ready for training a model: the
  images are converted to tensors of floats in the range [0, 1] and the labels
  are one-hot encoded.

  Args:
    dataset_name: Name of the dataset to load.
    batch_size: Batch size to be used for training.
    split: Split of the dataset that should be loaded.
    shuffle: Whether to shuffle the dataset.
    buffer_size: Size of the shuffle buffer.
    cache: Whether to cache the dataset after pre-processing.

  Returns:
    The batched pre-processed dataset.
  """
  dataset = tfds.load(dataset_name, split=split)
  max_label = dataset.reduce(
      np.int64(0), lambda state, x: tf.maximum(state, x['label']))
  num_labels = int(max_label) + 1
  dataset = dataset.map(
      functools.partial(_preprocess_image_dataset, num_labels=num_labels))
  if cache:
    dataset = dataset.cache()
  if shuffle:
    dataset = dataset.shuffle(buffer_size)

  return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
