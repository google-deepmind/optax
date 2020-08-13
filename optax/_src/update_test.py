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
"""Tests for `update.py`."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src import update


class UpdateTest(chex.TestCase):

  @chex.all_variants()
  def test_apply_updates(self):
    params = ({'a': jnp.ones((3, 2))}, jnp.ones((1,)))
    grads = jax.tree_map(lambda t: 2 * t, params)
    exp_params = jax.tree_map(lambda t: 3 * t, params)
    actual_params = self.variant(update.apply_updates)(params, grads)

    chex.assert_tree_all_close(
        exp_params, actual_params, atol=1e-10, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
