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
"""Tests for `transform.py`."""

from absl.testing import absltest
import jax.numpy as jnp

from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update
from optax._src import utils


STEPS = 50
LR = 1e-2


class TransformTest(absltest.TestCase):

  def setUp(self):
    super(TransformTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  def test_apply_every(self):
    # The frequency of the application of sgd
    k = 4
    zero_update = (jnp.array([0., 0.]), jnp.array([0., 0.]))

    # optax sgd
    optax_sgd_params = self.init_params
    sgd = alias.sgd(LR, 0.0)
    state_sgd = sgd.init(optax_sgd_params)

    # optax sgd plus apply every
    optax_sgd_apply_every_params = self.init_params
    sgd_apply_every = combine.chain(
        transform.apply_every(k=k), transform.trace(decay=0, nesterov=False),
        transform.scale(-LR))
    state_sgd_apply_every = sgd_apply_every.init(optax_sgd_apply_every_params)
    for i in range(STEPS):
      # Apply a step of sgd
      updates_sgd, state_sgd = sgd.update(self.per_step_updates, state_sgd)
      optax_sgd_params = update.apply_updates(optax_sgd_params, updates_sgd)

      # Apply a step of sgd_apply_every
      updates_sgd_apply_every, state_sgd_apply_every = sgd_apply_every.update(
          self.per_step_updates, state_sgd_apply_every)
      optax_sgd_apply_every_params = update.apply_updates(
          optax_sgd_apply_every_params, updates_sgd_apply_every)
      if i % k == k-1:
        # Check equivalence.
        utils.tree_all_close_assert(
            optax_sgd_apply_every_params, optax_sgd_params,
            atol=1e-6, rtol=1e-5)
      else:
        # Check update is zero.
        utils.tree_all_close_assert(
            updates_sgd_apply_every, zero_update, atol=0.0, rtol=0.0)


if __name__ == '__main__':
  absltest.main()
