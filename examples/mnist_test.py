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
"""Tests for optax.examples.mnist."""

from absl.testing import absltest
from absl.testing.absltest import mock

import chex
import haiku as hk
import jax
import numpy as np
import optax
import tensorflow as tf

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
import mnist  # Located in the examples folder.
# pylint: enable=g-bad-import-order


class MnistTest(chex.TestCase):

  def test_model_accuracy_returns_correct_result(self):
    """Checks that model_accuracy returns the correct result."""
    dataset = (
        dict(
            image=np.array([[-1, 0, 0], [-1, 0, 1]]),
            label=np.array([[1, 0, 0], [0, 1, 0]])),
        dict(
            image=np.array([[2, 2, 1], [-2, -1, 0]]),
            label=np.array([[0, 0, 1], [1, 0, 0]])),
    )
    self.assertEqual(
        mnist.model_accuracy(model=lambda x: -x, dataset=dataset), 0.75)

  @chex.all_variants
  def test_build_model_returns_mlp_with_correct_number_of_parameters(self):
    """Checks that the MLP has the correct number of parameters."""
    model = mnist.build_model(layer_dims=(4, 5))
    params = self.variant(model.init)(jax.random.PRNGKey(1), np.ones((9, 3, 2)))
    self.assertEqual(
        hk.data_structures.tree_size(params), (3 * 2 + 1) * 4 + (4 + 1) * 5)

  @chex.all_variants
  def test_build_model_returns_mlp_with_correct_output_shape(self):
    """Checks that the MLP has the correct output shape."""
    model = mnist.build_model(layer_dims=(4, 5))
    inputs = np.ones((9, 3, 2))
    params = model.init(jax.random.PRNGKey(1), inputs)
    outputs = self.variant(model.apply)(params, inputs)
    self.assertEqual(outputs.shape, (9, 5))

  def test_train_on_mnist_can_fit_linear_mock_data(self):
    """Checks that train_on_mnist can fit linear mock data."""
    data = {
        'image': np.arange(-0.5, 0.5, 0.1).reshape(10, 1, 1),
        'label': np.array([[1, 0]] * 4 + [[0, 1]] * 6)
    }
    dataset = tf.data.Dataset.from_tensor_slices(data).repeat(8).batch(10)
    with mock.patch.object(
        datasets, 'load_image_dataset', return_value=dataset):
      final_accuracy = mnist.train_on_mnist(optax.adam(0.01), hidden_sizes=(1,))

    self.assertEqual(final_accuracy, 1.)


if __name__ == '__main__':
  absltest.main()
