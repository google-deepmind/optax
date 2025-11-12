# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Utilities for optax tests."""

import contextlib
import logging
import re
import threading

import jax
import numpy as np

_LOG_LIST = []


class _LogsToListHandler(logging.Handler):
  """A handler for capturing logs programmatically without printing them."""

  def emit(self, record):
    _LOG_LIST.append(record)


logger = logging.getLogger("jax")
logger.addHandler(_LogsToListHandler())

# We need a lock to be able to run this context manager from multiple threads.
_compilation_log_lock = threading.Lock()


@contextlib.contextmanager
def log_compilations():
  """A utility for programmatically capturing JAX compilation logs."""
  with _compilation_log_lock, jax.log_compiles():
    _LOG_LIST.clear()
    compilation_logs = []
    yield compilation_logs  # these will contain the compilation logs
    compilation_logs.extend([
        log for log in _LOG_LIST
        if re.search(r"Finished .* compilation", log.getMessage())
    ])


def assert_trees_all_close(actual, desired, rtol=1e-6, atol=0.0, err_msg=None):
  """Asserts that two pytrees of arrays are close within a tolerance."""
  flat_a, tree_def_a = jax.tree_util.tree_flatten(actual)
  flat_d, tree_def_d = jax.tree_util.tree_flatten(desired)
  if tree_def_a != tree_def_d:
    raise AssertionError(
        f"Trees have different structures:\n{tree_def_a}\n{tree_def_d}"
    )
  for x, y in zip(flat_a, flat_d):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)


def assert_trees_all_equal(actual, desired, err_msg=None):
  """Asserts that two pytrees of arrays are equal."""
  flat_a, tree_def_a = jax.tree_util.tree_flatten(actual)
  flat_d, tree_def_d = jax.tree_util.tree_flatten(desired)
  if tree_def_a != tree_def_d:
    raise AssertionError(
        f"Trees have different structures:\n{tree_def_a}\n{tree_def_d}"
    )
  for x, y in zip(flat_a, flat_d):
    np.testing.assert_array_equal(x, y, err_msg=err_msg)
