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
"""Regression tests for fully ignored ranking entries."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses import _ranking


class RankingMaskTest(parameterized.TestCase):

  @parameterized.product(
      invalid=(np.nan, np.inf, -np.inf, 100.0),
      reduction=('mean', 'sum', 'none'),
      compiled=(False, True),
  )
  def test_masked_weights_do_not_poison_gradients(
      self, invalid, reduction, compiled
  ):
    scores = jnp.array([[2.0, 1.0, invalid], [0.5, 1.5, invalid]])
    labels = jnp.array([[1.0, 0.0, invalid], [0.0, 1.0, invalid]])
    weights = jnp.array([[2.0, 1.0, invalid], [1.0, 3.0, invalid]])
    mask = jnp.array([[True, True, False], [True, True, False]])
    reduce_fn = {'mean': jnp.mean, 'sum': jnp.sum, 'none': None}[reduction]

    def objective(x, y, w):
      return jnp.sum(
          _ranking.ranking_softmax_loss(
              x, y, where=mask, weights=w, reduce_fn=reduce_fn
          )
      )

    def reference(x, y, w):
      return jnp.sum(
          _ranking.ranking_softmax_loss(
              x[:, :2], y[:, :2], weights=w[:, :2], reduce_fn=reduce_fn
          )
      )

    evaluate = jax.value_and_grad(objective, argnums=(0, 1, 2))
    if compiled:
      evaluate = jax.jit(evaluate)
    actual, gradients = evaluate(scores, labels, weights)
    expected, reference_grads = jax.value_and_grad(
        reference, argnums=(0, 1, 2)
    )(scores, labels, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    for gradient, reference_grad in zip(gradients, reference_grads):
      np.testing.assert_allclose(gradient, reference_grad, rtol=1e-6, atol=1e-7)
      np.testing.assert_array_equal(gradient[:, -1], 0.0)

  def test_all_masked_nonfinite_inputs_have_zero_loss_and_gradients(self):
    x = jnp.array([jnp.nan, jnp.inf])
    mask = jnp.array([False, False])
    value, gradients = jax.jit(
        jax.value_and_grad(
            lambda x, y, w: _ranking.ranking_softmax_loss(
                x, y, weights=w, where=mask
            ),
            argnums=(0, 1, 2),
        )
    )(x, x, x)
    self.assertEqual(float(value), 0.0)
    for gradient in gradients:
      np.testing.assert_array_equal(gradient, 0.0)

  def test_invalid_unmasked_weights_are_not_hidden(self):
    result = _ranking.ranking_softmax_loss(
        jnp.array([1.0, 2.0]),
        jnp.array([1.0, 0.0]),
        weights=jnp.array([jnp.nan, 1.0]),
        where=jnp.array([True, True]),
    )
    self.assertTrue(np.isnan(result))


if __name__ == '__main__':
  absltest.main()
