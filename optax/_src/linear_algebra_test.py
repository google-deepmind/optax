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
"""Tests for optax._src.linear_algebra."""

from absl.testing import absltest

import jax.numpy as jnp
import numpy as np
from optax._src import linear_algebra
import scipy.stats


class LinearAlgebraTest(absltest.TestCase):

  def test_global_norm(self):
    flat_updates = jnp.array([2., 4., 3., 5.], dtype=jnp.float32)
    nested_updates = dict(
        a=jnp.array([2., 4.], dtype=jnp.float32),
        b=jnp.array([3., 5.], dtype=jnp.float32))
    np.testing.assert_array_equal(
        jnp.sqrt(jnp.sum(flat_updates**2)),
        linear_algebra.global_norm(nested_updates))

  def test_matrix_inverse_pth_root(self):
    """Test for matrix inverse pth root."""

    def _gen_symmetrix_matrix(dim, condition_number):
      u = scipy.stats.ortho_group.rvs(dim=dim).astype(np.float64)
      v = u.T
      diag = np.diag([condition_number ** (-i/(dim-1)) for i in range(dim)])
      return u @ diag @ v

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10 ** e
      ms = _gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number),
          condition_number * 0.01)
      error = linear_algebra.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12)[1]
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass


if __name__ == '__main__':
  absltest.main()
