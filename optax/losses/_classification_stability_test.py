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
"""Integer-label cross entropy should be invariant to a logit offset."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses import _classification


class IntegerCrossEntropyStabilityTest(parameterized.TestCase):

  @parameterized.product(
      dtype_and_offset=(
          (jnp.float32, 1e6),
          (jnp.float32, 1e8),
          (jnp.float16, 4096),
          (jnp.bfloat16, 256),
      ),
      sign=(-1, 1),
  )
  def test_uniform_logits_keep_nonzero_loss(self, dtype_and_offset, sign):
    dtype, offset = dtype_and_offset
    logits = jnp.full((2, 3), sign * offset, dtype=dtype)
    labels = jnp.array([0, 2])
    fn = _classification.softmax_cross_entropy_with_integer_labels
    expected = jnp.full((2,), np.log(3), dtype=dtype)
    for loss in (fn, jax.jit(fn)):
      actual = loss(logits, labels)
      self.assertEqual(actual.dtype, logits.dtype)
      np.testing.assert_allclose(
          actual, expected, rtol=2 * jnp.finfo(dtype).eps
      )

  @parameterized.parameters(
      {"axis": -1}, {"axis": 1}, {"axis": (1, 2)}, {"axis": (2, 1)}
  )
  def test_nonuniform_loss_and_gradient_preserve_shift(self, axis):
    logits = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 8
    axes = (axis,) if isinstance(axis, int) else axis
    axes = tuple(a % logits.ndim for a in axes)
    batch_shape = tuple(s for i, s in enumerate(logits.shape) if i not in axes)
    labels = jnp.ones(batch_shape, dtype=jnp.int32)
    fn = lambda x: _classification.softmax_cross_entropy_with_integer_labels(
        x, labels, axis=axis
    )
    shifted = logits + 1e6
    for loss in (fn, jax.jit(fn)):
      np.testing.assert_allclose(loss(shifted), loss(logits), rtol=1e-6)
    gradient = jax.jit(jax.grad(lambda x: fn(x).sum()))
    np.testing.assert_allclose(gradient(shifted), gradient(logits), atol=1e-7)

  def test_masked_large_logit_does_not_set_the_shift(self):
    logits = jnp.array([[0.0, 0.0, 1e8]])
    fn = lambda x: _classification.softmax_cross_entropy_with_integer_labels(
        x, jnp.array([0]), where=jnp.array([[True, True, False]])
    )
    np.testing.assert_allclose(jax.jit(fn)(logits), [np.log(2)], rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda x: fn(x).sum())(logits), [[-0.5, 0.5, 0.0]]
    )

  def test_all_masked_rows_keep_zero_loss_and_gradient(self):
    logits = jnp.full((2, 3), 1e8)
    fn = lambda x: _classification.softmax_cross_entropy_with_integer_labels(
        x, jnp.array([0, 1]), where=jnp.array([False, False])
    )
    np.testing.assert_array_equal(jax.jit(fn)(logits), [0.0, 0.0])
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: fn(x).sum()))(logits), jnp.zeros_like(logits)
    )

  def test_negative_infinite_non_target_logit(self):
    logits = jnp.array([[1e8, -jnp.inf, 1e8]])
    actual = _classification.softmax_cross_entropy_with_integer_labels(
        logits, jnp.array([0])
    )
    np.testing.assert_allclose(actual, [np.log(2)], rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
