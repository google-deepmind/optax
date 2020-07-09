# Lint as: python3
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
"""Utility functions for testing."""

import collections
import functools
import jax
import numpy as np


def equal_shape_assert(inputs):
  """Checks that all arrays have the same shape.

  Args:
    inputs: sequence of arrays.

  Raises:
    AssertionError: if the shapes of all arrays do not match.
  """
  if isinstance(inputs, collections.Sequence):
    shape = inputs[0].shape
    expected_shapes = [shape] * len(inputs)
    shapes = [x.shape for x in inputs]
    if shapes != expected_shapes:
      raise AssertionError(f"Arrays have different shapes: {shapes}.")


def tree_all_close_assert(
    actual, desired, rtol: float = 1e-07, atol: float = 0):
  """Assert two trees have leaves with approximately equal values.

  This compares the difference between values of actual and desired to
   atol + rtol * abs(desired).

  Args:
    actual: pytree with array leaves.
    desired: pytree with array leaves.
    rtol: relative tolerance.
    atol: absolute tolerance.
  Raise:
    AssertionError: if the leaf values actual and desired are not equal up to
      specified tolerance.
  """
  if jax.tree_structure(actual) != jax.tree_structure(desired):
    raise AssertionError(
        "Error in value equality check: Trees do not have the same structure,\n"
        f"actual: {jax.tree_structure(actual)}\n"
        f"desired: {jax.tree_structure(desired)}.")

  assert_fn = functools.partial(
      np.testing.assert_allclose,
      rtol=rtol,
      atol=atol,
      err_msg="Error in value equality check: Values not approximately equal")
  jax.tree_multimap(assert_fn, actual, desired)
