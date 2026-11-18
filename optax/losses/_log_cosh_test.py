# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Numerical regressions for log-cosh near zero and across its branches."""

from decimal import Decimal
from decimal import localcontext

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax._src import utils


def _reference(values):
  """Evaluate the defining formula and analytic derivatives independently."""
  results = []
  with localcontext() as context:
    context.prec = 80
    for value in np.asarray(values).flat:
      x = Decimal.from_float(float(value))
      positive, negative = x.exp(), (-x).exp()
      total = positive + negative
      results.append(
          (
              float((total / 2).ln()),
              float((positive - negative) / total),
              float(4 / total**2),
          )
      )
  return np.asarray(results).T.reshape((3,) + np.shape(values))


class LogCoshNumericsTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float32, jnp.float64),
      compiled=(False, True),
      with_targets=(False, True),
  )
  def test_small_nonzero_residuals(self, dtype, compiled, with_targets):
    with utils.x64_precision(dtype == jnp.float64):
      small, medium = (1e-8, 1e-7) if dtype == jnp.float64 else (1e-4, 1e-3)
      predictions = jnp.asarray(
          [0.0, small, -small, medium, -medium, 0.5, -0.5], dtype=dtype
      )
      targets = jnp.zeros((1, 1), dtype=dtype) if with_targets else None
      function = lambda p: optax.log_cosh(p, targets)
      if compiled:
        function = jax.jit(function)
      actual = function(predictions)
      residuals = predictions if targets is None else predictions - targets
      expected = _reference(residuals)[0]
      self.assertEqual(actual.shape, residuals.shape)
      self.assertEqual(actual.dtype, dtype)
      self.assertTrue(np.isfinite(actual).all())
      self.assertTrue((np.asarray(actual) >= 0).all())
      self.assertEqual(float(np.asarray(actual).flat[0]), 0.0)
      np.testing.assert_allclose(
          actual, expected, rtol=1e-12 if dtype == jnp.float64 else 3e-6, atol=0
      )

  @parameterized.product(
      dtype=(jnp.float32, jnp.float64),
      compiled=(False, True),
      with_targets=(False, True),
  )
  def test_branch_values_and_derivatives(self, dtype, compiled, with_targets):
    with utils.x64_precision(dtype == jnp.float64):
      predictions = jnp.asarray(
          [-700, -20, -1.01, -1, -0.99, -0.5, 0, 0.5, 0.99, 1, 1.01, 20, 700],
          dtype=dtype,
      )
      target = jnp.asarray(0.0, dtype=dtype) if with_targets else None
      scalar = lambda p: optax.log_cosh(p, target)
      first, second = jax.grad(scalar), jax.grad(jax.grad(scalar))
      function = lambda p: tuple(
          jax.vmap(f)(p) for f in (scalar, first, second)
      )
      if compiled:
        function = jax.jit(function)
      actual = function(predictions)
      expected = _reference(predictions)
      tolerance = 1e-12 if dtype == jnp.float64 else 3e-6
      for result, reference in zip(actual, expected):
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.shape, predictions.shape)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(
            result, reference, rtol=tolerance, atol=tolerance
        )


if __name__ == '__main__':
  absltest.main()
