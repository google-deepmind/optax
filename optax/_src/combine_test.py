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
"""Tests for `combine.py`."""

from absl.testing import absltest

import chex
import jax.numpy as jnp

from optax._src import combine
from optax._src import transform
from optax._src import update


STEPS = 50
LR = 1e-2


class ComposeTest(chex.TestCase):

  def setUp(self):
    super(ComposeTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants()
  def test_chain(self):
    transformations = [
        transform.scale_by_adam(),
        transform.trace(decay=0, nesterov=False),
        transform.scale(-LR)]

    # Apply updates with chain.
    chain_params = self.init_params
    chained_transforms = combine.chain(*transformations)
    state = chained_transforms.init(chain_params)

    @self.variant
    def update_fn(updates, state):
      return chained_transforms.update(updates, state)

    for _ in range(STEPS):
      updates, state = update_fn(self.per_step_updates, state)
      chain_params = update.apply_updates(chain_params, updates)

    # Manually apply sequence of transformations.
    manual_params = self.init_params
    states = [t.init(manual_params) for t in transformations]
    for _ in range(STEPS):
      updates = self.per_step_updates
      new_states = []
      for t, s in zip(transformations, states):
        updates, state = t.update(updates, s)
        new_states.append(state)
      manual_params = update.apply_updates(manual_params, updates)
      states = new_states

    # Check equivalence.
    chex.assert_tree_all_close(manual_params, chain_params, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
