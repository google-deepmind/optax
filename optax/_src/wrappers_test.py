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
"""Tests for `wrappers.py`."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
from optax._src import alias
from optax._src import update
from optax._src import wrappers


class WrappersTest(chex.TestCase):

  def test_flatten(self):
    def init_params():
      return (jnp.array([1., 2.]), jnp.array([3., 4.]))

    per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

    # First calculate new params without flattening
    optax_sgd_params = init_params()
    sgd = alias.sgd(1e-2, 0.0)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, state_sgd = sgd.update(per_step_updates, state_sgd)
    sgd_params_no_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # And now calculate new params with flattening
    optax_sgd_params = init_params()
    sgd = wrappers.flatten(sgd)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, state_sgd = sgd.update(per_step_updates, state_sgd)
    sgd_params_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # Test that both give the same result
    chex.assert_tree_all_close(
        sgd_params_no_flatten, sgd_params_flatten, atol=1e-7, rtol=1e-7)


if __name__ == '__main__':
  absltest.main()
