# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `utils.py`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import jax

from optax._src import utils


class ScaleGradientTest(parameterized.TestCase):

  @parameterized.product(inputs=[-1., 0., 1.], scale=[-0.5, 0., 0.5, 1., 2.])
  @mock.patch.object(jax.lax, 'stop_gradient', wraps=jax.lax.stop_gradient)
  def test_scale_gradient(self, mock_sg, inputs, scale):

    def fn(inputs):
      outputs = utils.scale_gradient(inputs, scale)
      return outputs ** 2

    grad = jax.grad(fn)
    self.assertEqual(grad(inputs), 2 * inputs * scale)
    if scale == 0.:
      mock_sg.assert_called_once_with(inputs)
    else:
      self.assertFalse(mock_sg.called)
    self.assertEqual(fn(inputs), inputs ** 2)

  @parameterized.product(scale=[-0.5, 0., 0.5, 1., 2.])
  def test_scale_gradient_pytree(self, scale):

    def fn(inputs):
      outputs = utils.scale_gradient(inputs, scale)
      outputs = jax.tree_util.tree_map(lambda x: x ** 2, outputs)
      return sum(jax.tree_util.tree_leaves(outputs))

    inputs = dict(a=-1., b=dict(c=(2.,), d=0.))

    grad = jax.grad(fn)
    grads = grad(inputs)
    jax.tree_util.tree_map(
        lambda i, g: self.assertEqual(g, 2 * i * scale), inputs, grads)
    self.assertEqual(
        fn(inputs),
        sum(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: x**2, inputs))))

if __name__ == '__main__':
  absltest.main()
