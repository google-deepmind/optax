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
"""Tests for numerical methods in `numerics.py`."""

import functools
import itertools
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import numerics
from optax._src import transform

_ALL_ORDS = [None, np.inf, -np.inf, "fro", "nuc", 0, 1, 2, -2, -2, -1.5, 1.5]


def int32_array(i):
  return jnp.array(i, dtype=jnp.int32)


def float32_array(i):
  return jnp.array(i, dtype=jnp.float32)


def _invalid_ord_axis_inputs(ord_axis_keepdims):
  ord_, axis = ord_axis_keepdims[0], ord_axis_keepdims[1]
  return any((
      (ord_ == 0 and axis is None),
      (isinstance(ord_, float) and axis is None),
      (isinstance(ord_, str) and axis is not None),
  ))


class NumericsTest(parameterized.TestCase):

  @parameterized.parameters(*(
      "bfloat16",
      "float16",
      "float32",
      "float64",
      "int8",
      "int16",
      "int32",
      "int64",
      "uint8",
      "uint16",
      "uint32",
      "uint64",
  ))
  def test_safe_increment(self, dtype):
    """Tests that safe_increment works for all dtypes."""
    if dtype in ["float64", "int64", "uint64"]:
      jax.config.update("jax_enable_x64", True)
    dtype = jnp.dtype(dtype)
    inc_fn = jax.jit(numerics.safe_increment)

    with self.subTest("Increments correctly"):
      base = jnp.asarray(3, dtype=dtype)
      incremented = inc_fn(base)
      expected = jnp.asarray(4, dtype=dtype)
      np.testing.assert_array_equal(incremented, expected)

    with self.subTest("Avoids overflow"):
      if jnp.issubdtype(dtype, jnp.integer):
        max_val = jnp.iinfo(dtype).max
      elif jnp.issubdtype(dtype, jnp.floating):
        max_val = jnp.finfo(dtype).max
      else:
        raise ValueError(f"Unsupported dtype: {dtype}")
      base = jnp.asarray(max_val, dtype=dtype)
      incremented = inc_fn(base)
      np.testing.assert_array_equal(incremented, base)
    if dtype in ["float64", "int64", "uint64"]:
      jax.config.update("jax_enable_x64", False)

  @parameterized.product(
      str_dtype=[
          "bool",
          "complex64",
      ]
  )
  def test_safe_increment_exceptions(self, str_dtype):
    """Tests that safe_increment raises exceptions for invalid dtypes."""
    dtype = jnp.dtype(str_dtype)
    self.assertRaises(
        ValueError, numerics.safe_increment, jnp.asarray(1, dtype=dtype)
    )

  @parameterized.parameters(
      itertools.filterfalse(
          _invalid_ord_axis_inputs,
          itertools.product(_ALL_ORDS, [None, 0, 1], [False, True]),
      )
  )
  def test_safe_norm(self, ord, axis, keepdims):  # pylint: disable=redefined-builtin
    dnorm_dx = jax.jit(
        jax.jacfwd(
            functools.partial(
                numerics.safe_norm, ord=ord, axis=axis, keepdims=keepdims
            ),
            argnums=0,
        )
    )
    # Test gradient is 0. in 0. when zero min norm is used.
    g = dnorm_dx(float32_array(jnp.zeros((3, 4))), float32_array(0.0))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))
    # Test gradient is 0. in 0. when non zero min norm is used.
    g = dnorm_dx(float32_array(jnp.zeros((3, 4))), float32_array(3.0))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))

  def test_safe_rms(self):
    drms_dx = jax.jit(jax.grad(numerics.safe_root_mean_squares))
    # Test gradient is 0. in 0. when zero min rms is used.
    g = drms_dx(float32_array(0.0), float32_array(0.0))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))
    # Test gradient is 0. in 0. when non zero min rms is used.
    g = drms_dx(float32_array(0.0), float32_array(3.0))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))

  @parameterized.product(
      values=[[300.0, 400.0], [1e-5, 2e-5], [0.0, 0.0]],
      min_rms=[0.0, 1e-3],
      compiled=[False, True],
  )
  def test_safe_rms_float16_value_and_gradient(self, values, min_rms, compiled):
    x = jnp.array(values, dtype=jnp.float16)
    fn = jax.value_and_grad(
        lambda x: numerics.safe_root_mean_squares(x, min_rms)
    )
    if compiled:
      fn = jax.jit(fn)
    value, gradient = fn(x)
    reference = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(reference**2))
    expected_gradient = (
        reference / (reference.size * rms)
        if rms > min_rms else np.zeros_like(reference)
    )
    self.assertEqual(value.dtype, x.dtype)
    self.assertEqual(gradient.dtype, x.dtype)
    np.testing.assert_allclose(value, max(rms, min_rms), rtol=3e-3, atol=3e-8)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-3)

  @parameterized.parameters(jnp.bfloat16, jnp.float32, jnp.complex64)
  def test_safe_rms_preserves_result_dtype(self, dtype):
    x = jnp.array([3.0, 4.0], dtype=dtype)
    result = jax.jit(numerics.safe_root_mean_squares)(x, 0.0)
    self.assertEqual(result.dtype, jnp.real(x).dtype)
    np.testing.assert_allclose(result, np.sqrt(12.5), rtol=1e-2)
    result = jax.jit(numerics.safe_root_mean_squares)(
        x, jnp.asarray(5.0, dtype=jnp.float32)
    )
    self.assertEqual(result.dtype, jnp.float32)
    np.testing.assert_allclose(result, 5.0)

  @parameterized.parameters(False, True)
  def test_param_block_rms_float16(self, compiled):
    params = jnp.array([300.0, 400.0], dtype=jnp.float16)
    updates = jnp.full_like(params, 1e-3)
    tx = transform.scale_by_param_block_rms()
    update = jax.jit(tx.update) if compiled else tx.update
    result, _ = update(updates, tx.init(params), params)
    expected = np.asarray(updates, dtype=np.float64) * np.sqrt(125_000.0)
    self.assertEqual(result.dtype, updates.dtype)
    np.testing.assert_allclose(result, expected, rtol=3e-3)

  @parameterized.product(
      sq_fn=[lambda x: x**2, lambda x: x * x, jnp.square],
      x=[
          3,
          3.0,
          np.array([4, 5.2]),
          1j,
          3.0 + 1j,
          np.array([4 + 1j, 5.2 + 1j]),
      ],
  )
  def test_abs_sqr_hlo_equivalence(self, sq_fn, x):
    # Tests that JAX generates the same HLO from `numerics.abs_sq`,
    # `jnp.square(x)`, `x * x`,  and `x**2` if they are real
    # Check that it does not generate the same hlo if they are complex.

    def _get_hlo_repr(f, x):
      hlo_string = jax.jit(f).lower(x).compiler_ir(dialect="hlo").as_hlo_text()
      hlo_string = re.search(r"ENTRY.*", hlo_string, re.DOTALL)
      if hlo_string:
        hlo_string = hlo_string.group()
      else:
        raise ValueError(f"No ENTRY found in HLO string: {hlo_string}")
      hlo_string = re.sub("ENTRY.*?{", "ENTRY XXXX", hlo_string)
      hlo_string = re.sub("HloModule.*?\n", "", hlo_string)
      hlo_string = re.sub("ROOT.*?=", "ROOT result.2 =", hlo_string)
      return hlo_string

    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
      self.assertNotEqual(
          _get_hlo_repr(sq_fn, x),
          _get_hlo_repr(numerics.abs_sq, x),
      )
    else:
      self.assertEqual(
          _get_hlo_repr(sq_fn, x),
          _get_hlo_repr(numerics.abs_sq, x),
      )


if __name__ == "__main__":
  absltest.main()
