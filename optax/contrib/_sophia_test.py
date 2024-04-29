# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `sophia.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax.contrib import _sophia


class SophiaTest(chex.TestCase):

  @chex.all_variants
  @parameterized.product(
      [
          dict(
              params=(np.array([1.0, -77.0]), np.array([3.0, 1.12])),
              updates=(np.array([500.0, -400.0]), np.array([300.0, 340.0])),
          ),
          dict(
              params=dict(a=np.array([7.0, -77.0]), b=np.array([[[112.112]]])),
              updates=dict(a=np.array([5e-7, -4e-7]), b=np.array([[1e8]])),
          ),
      ],
      [
          dict(learning_rate=1e-4),
          dict(learning_rate=1e-2),
      ],
  )
  def test_sophia(self, params, updates, learning_rate):
    params = jax.tree_util.tree_map(jnp.asarray, params)
    updates = jax.tree_util.tree_map(jnp.asarray, updates)

    optim = _sophia.sophia(learning_rate)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    updates, state = transform_fn(updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))
    jax.tree_util.tree_map(
        lambda *args: chex.assert_equal_shape(args), params, updates
    )


if __name__ == "__main__":
  absltest.main()
