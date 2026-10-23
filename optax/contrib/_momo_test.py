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
"""Doctest coverage for the MoMo examples."""

import doctest

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax
from optax import contrib as _contrib
from optax.contrib import _momo


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
        _momo,
        globs={
          "jax": jax,
          "jnp": jnp,
          "optax": optax,
          "contrib": _contrib,
        },
      )
  )
  return tests


if __name__ == "__main__":
  absltest.main()
