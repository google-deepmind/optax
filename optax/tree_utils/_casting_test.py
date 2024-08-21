# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for optax.tree_utils._casting."""

from absl.testing import absltest
from absl.testing import parameterized

from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np

from optax import tree_utils as otu


class CastingTest(parameterized.TestCase):

  @parameterized.parameters([
      (jnp.float32, [1.3, 2.001, 3.6], [-3.3], [1.3, 2.001, 3.6], [-3.3]),
      (jnp.float32, [1.3, 2.001, 3.6], [-3], [1.3, 2.001, 3.6], [-3.0]),
      (jnp.int32, [1.3, 2.001, 3.6], [-3.3], [1, 2, 3], [-3]),
      (jnp.int32, [1.3, 2.001, 3.6], [-3], [1, 2, 3], [-3]),
      (None, [1.123, 2.33], [0.0], [1.123, 2.33], [0.0]),
      (None, [1, 2, 3], [0.0], [1, 2, 3], [0.0]),
  ])
  def test_tree_cast(self, dtype, b, c, new_b, new_c):
    def _build_tree(val1, val2):
      dict_tree = {'a': {'b': jnp.array(val1)}, 'c': jnp.array(val2)}
      return jtu.tree_map(lambda x: x, dict_tree)

    tree = _build_tree(b, c)
    tree = otu.tree_cast(tree, dtype=dtype)
    jtu.tree_map(
        np.testing.assert_array_equal, tree, _build_tree(new_b, new_c)
    )


if __name__ == '__main__':
  absltest.main()
