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
