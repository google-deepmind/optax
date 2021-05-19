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
"""Tests for optax._src.numerics."""

from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import numerics


int32_array = lambda i: jnp.array(i, dtype=jnp.int32)
float32_array = lambda i: jnp.array(i, dtype=jnp.float32)


class NumericsTest(chex.TestCase):

  @chex.all_variants()
  def test_safe_int32_increments(self):
    inc_fn = self.variant(numerics.safe_int32_increment)
    # increment small numbers correctly.
    base = int32_array(3)
    incremented = inc_fn(base)
    np.testing.assert_array_equal(incremented, int32_array(4))
    # avoid overflow when incrementing maxint.
    base = int32_array(np.iinfo(np.int32).max)
    incremented = inc_fn(base)
    np.testing.assert_array_equal(incremented, base)

  @chex.all_variants()
  def test_safe_norm(self):
    dnorm_dx = self.variant(jax.grad(numerics.safe_norm))
    # Test gradient is 0. in 0. when zero min norm is used.
    g = dnorm_dx(float32_array(0.), float32_array(0.))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))
    # Test gradient is 0. in 0. when non zero min norm is used.
    g = dnorm_dx(float32_array(0.), float32_array(3.))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))

  @chex.all_variants()
  def test_safe_rms(self):
    drms_dx = self.variant(jax.grad(numerics.safe_root_mean_squares))
    # Test gradient is 0. in 0. when zero min rms is used.
    g = drms_dx(float32_array(0.), float32_array(0.))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))
    # Test gradient is 0. in 0. when non zero min rms is used.
    g = drms_dx(float32_array(0.), float32_array(3.))
    np.testing.assert_array_equal(g, jnp.zeros_like(g))


if __name__ == '__main__':
  absltest.main()
