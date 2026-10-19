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
"""Tests for regression losses in `optax.losses._regression.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import update
from optax.losses import _regression


class SquaredErrorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([-2.0, -1.0, 0.5, 1.0])
    self.ts = jnp.array([-1.5, 0.0, -1, 1.0])
    # compute expected outputs in numpy.
    self.exp = (self.ts - self.ys) ** 2

  def test_scalar(self):
    np.testing.assert_allclose(
        jax.jit(_regression.squared_error)(self.ys[0], self.ts[0]),
        self.exp[0],
    )

  def test_batched(self):
    np.testing.assert_allclose(
        jax.jit(_regression.squared_error)(self.ys, self.ts), self.exp
    )

  def test_shape_mismatch(self):
    with self.assertRaises(ValueError):
      _ = jax.jit(_regression.squared_error)(
          self.ys, jnp.expand_dims(self.ts, axis=-1)
      )


class L2LossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([-2.0, -1.0, 0.5, 1.0])
    self.ts = jnp.array([-1.5, 0.0, -1, 1.0])
    # compute expected outputs in numpy.
    self.exp = 0.5 * (self.ts - self.ys) ** 2

  def test_scalar(self):
    np.testing.assert_allclose(
        jax.jit(_regression.l2_loss)(self.ys[0], self.ts[0]), self.exp[0]
    )

  def test_batched(self):
    np.testing.assert_allclose(
        jax.jit(_regression.l2_loss)(self.ys, self.ts), self.exp
    )

  def test_shape_mismatch(self):
    with self.assertRaises(ValueError):
      _ = jax.jit(_regression.l2_loss)(
          self.ys, jnp.expand_dims(self.ts, axis=-1)
      )


class HuberLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([-2.0, 0.5, 0.0, 0.5, 2.0, 4.0, 132.0])
    self.ts = np.array([0.0, -0.5, 0.0, 1.0, 1.0, 2.0, 0.3])
    # computed expected outputs manually.
    self.exp = np.array([1.5, 0.5, 0.0, 0.125, 0.5, 1.5, 131.2])

  def test_scalar(self):
    np.testing.assert_allclose(
        jax.jit(_regression.huber_loss)(self.ys[0], self.ts[0], delta=1.0),
        self.exp[0],
    )

  def test_batched(self):
    np.testing.assert_allclose(
        jax.jit(_regression.huber_loss)(self.ys, self.ts, delta=1.0),
        self.exp,
    )


# TODO(b/188419459): add test for grad and second order grad.
class LogCoshTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Test large values for overflow
    self.ys = jnp.array([500, -2.0, -1.0, 0.5, 1.0])
    self.ts = jnp.array([-200, -1.5, 0.0, -1, 1.0])
    # computed using tensorflow.keras.losses.log_cosh v2.4.1
    self.exp = jnp.array([699.3068, 0.12011445, 0.4337809, 0.85544014, 0.0])
    self.exp_ys_only = jnp.array(
        [499.30685, 1.3250027, 0.4337809, 0.12011451, 0.43378082]
    )

  def test_scalar(self):
    out = jax.jit(_regression.log_cosh)(self.ys[0], self.ts[0])
    np.testing.assert_allclose(out, self.exp[0], atol=1e-5)

  def test_batched(self):
    out = jax.jit(_regression.log_cosh)(self.ys, self.ts)
    np.testing.assert_allclose(out, self.exp, atol=1e-5)

  def test_scalar_predictions_only(self):
    out = jax.jit(_regression.log_cosh)(self.ys[0])
    np.testing.assert_allclose(out, self.exp_ys_only[0], atol=1e-5)

  def test_batched_predictions_only(self):
    out = jax.jit(_regression.log_cosh)(self.ys)
    np.testing.assert_allclose(out, self.exp_ys_only, atol=1e-5)


class CosineDistanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[10.0, 1.0, -2.0], [1.0, 4.0, 0.2]], dtype=np.float32)
    self.ts = np.array([[0.0, 1.2, 0.2], [1.0, -0.3, 0.0]], dtype=np.float32)
    # distance computed expected output from `scipy 1.20`.
    self.exp = np.array([0.9358251989, 1.0464068465], dtype=np.float32)

  def test_scalar_distance(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        jax.jit(_regression.cosine_distance)(self.ys[0], self.ts[0]),
        self.exp[0],
        atol=1e-4,
    )

  def test_scalar_similarity(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        jax.jit(_regression.cosine_similarity)(self.ys[0], self.ts[0]),
        1.0 - self.exp[0],
        atol=1e-4,
    )

  def test_batched_distance(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        jax.jit(_regression.cosine_distance)(self.ys, self.ts),
        self.exp,
        atol=1e-4,
    )

  def test_batched_similarity(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        jax.jit(_regression.cosine_similarity)(self.ys, self.ts),
        1.0 - self.exp,
        atol=1e-4,
    )

  @parameterized.parameters({'size': 5}, {'size': 10})
  def test_mask_distance(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.normal(size=size)
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _regression.cosine_distance(preds[mask], targets[mask])
    y = _regression.cosine_distance(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters({'size': 5}, {'size': 10})
  def test_mask_similarity(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.normal(size=size)
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _regression.cosine_similarity(preds[mask], targets[mask])
    y = _regression.cosine_similarity(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      {'axis': 0, 'shape': [4, 5, 6]},
      {'axis': 1, 'shape': [4, 5, 6]},
      {'axis': 2, 'shape': [4, 5, 6]},
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.normal(size=shape)
    x = _regression.cosine_similarity(preds, targets, axis=axis)
    y = _regression.cosine_similarity(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


def _reference(first, second, mask):
  a, b = np.asarray(first)[mask], np.asarray(second)[mask]
  norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
  cosine = np.dot(a, b) / (norm_a * norm_b)
  grad_a = np.zeros_like(first)
  grad_b = np.zeros_like(second)
  grad_a[mask] = b / (norm_a * norm_b) - cosine * a / norm_a**2
  grad_b[mask] = a / (norm_a * norm_b) - cosine * b / norm_b**2
  return cosine, (grad_a, grad_b)


class CosineMaskGradientTest(parameterized.TestCase):

  @parameterized.product(
      invalid=(np.nan, np.inf, -np.inf),
      operand=('first', 'second', 'both'),
      compiled=(False, True),
  )
  def test_excluded_values_have_zero_gradient_and_leave_valid_values_finite(
      self, invalid, operand, compiled
  ):
    first = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    second = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    if operand in ('first', 'both'):
      first[1] = invalid
    if operand in ('second', 'both'):
      second[1] = invalid
    mask = np.array([True, False, True])
    expected_value, expected_grads = _reference(first, second, mask)
    function = jax.value_and_grad(
        lambda a, b: _regression.cosine_similarity(a, b, where=mask),
        argnums=(0, 1),
    )
    if compiled:
      function = jax.jit(function)
    value, grads = function(jnp.asarray(first), jnp.asarray(second))
    np.testing.assert_allclose(value, expected_value, rtol=2e-6)
    for actual, expected in zip(grads, expected_grads):
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
      self.assertEqual(float(actual[1]), 0.0)

  @parameterized.parameters((-1,), (0,), (None,), ((0, 1),))
  def test_axes_and_broadcast_masks_match_explicit_zero_filling(self, axis):
    first = jnp.array([[1.0, np.nan, 2.0], [2.0, np.inf, 3.0]])
    second = jnp.array([[2.0, 6.0, 1.0], [3.0, 8.0, 2.0]])
    mask = jnp.array([[True, False, True]])
    clean_first = jnp.where(mask, first, 0.0)
    clean_second = jnp.where(mask, second, 0.0)
    fn = lambda a, b: _regression.cosine_similarity(
        a, b, axis=axis, where=mask, epsilon=1e-5
    ).sum()
    expected_fn = lambda a, b: _regression.cosine_similarity(
        a, b, axis=axis, epsilon=1e-5
    ).sum()
    value, grads = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(
        first, second
    )
    expected, expected_grads = jax.value_and_grad(expected_fn, argnums=(0, 1))(
        clean_first, clean_second
    )
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    for actual, reference in zip(grads, expected_grads):
      np.testing.assert_allclose(
          actual, jnp.where(mask, reference, 0.0), rtol=2e-6, atol=1e-7
      )

  def test_all_masked_inputs_respect_positive_epsilon(self):
    values = jnp.array([np.nan, np.inf])
    fn = lambda x: _regression.cosine_distance(
        x, x, where=jnp.array([False, False]), epsilon=1e-5
    )
    loss, grad = jax.jit(jax.value_and_grad(fn))(values)
    self.assertEqual(float(loss), 1.0)
    np.testing.assert_array_equal(grad, [0.0, 0.0])

  def test_included_nonfinite_values_are_not_silently_removed(self):
    first = jnp.array([1.0, jnp.nan, 2.0])
    second = jnp.array([2.0, 3.0, 1.0])
    for where in (
        None,
        jnp.array([True, True, True]),
        jnp.array([False, True, True]),
    ):
      with self.subTest(where=where):
        value = _regression.cosine_similarity(first, second, where=where)
        self.assertTrue(np.isnan(value))

  def test_masked_targets_allow_a_real_sgd_update(self):
    params = jnp.array([1.0, 7.0, 2.0], dtype=jnp.float32)
    targets = jnp.array([2.0, np.nan, 1.0], dtype=jnp.float32)
    mask = jnp.array([True, False, True])
    objective = lambda p: _regression.cosine_distance(p, targets, where=mask)
    optimizer = alias.sgd(0.1)
    state = optimizer.init(params)

    @jax.jit
    def step(p, s):
      loss, grad = jax.value_and_grad(objective)(p)
      updates, s = optimizer.update(grad, s, p)
      return update.apply_updates(p, updates), s, loss

    initial_loss = objective(params)
    for _ in range(5):
      params, state, loss = step(params, state)
      self.assertTrue(np.isfinite(params).all())
      self.assertTrue(np.isfinite(loss))
    self.assertLess(float(objective(params)), float(initial_loss))
    self.assertEqual(float(params[1]), 7.0)


if __name__ == '__main__':
  absltest.main()
