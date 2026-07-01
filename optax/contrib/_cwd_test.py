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
"""Tests for `optax.contrib._cwd`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from optax._src import test_utils
from optax.contrib import _cwd


class CautiousWeightDecayTest(parameterized.TestCase):

  def test_cautious_weight_decay_alignment(self):
    # Case 1: Signs align (decay applied)
    # Case 2: Signs differ (decay NOT applied)
    # Case 3: Zero update or zero param (handled as usual)

    params = {'x': jnp.array([1.0, -1.0, 1.0, -1.0, 0.0])}
    updates = {'x': jnp.array([0.5, -0.5, -0.5, 0.5, 1.0])}
    # Index 0: 1.0, 0.5 -> Aligned (+) -> Decay
    # Index 1: -1.0, -0.5 -> Aligned (-) -> Decay
    # Index 2: 1.0, -0.5 -> Misaligned -> No Decay
    # Index 3: -1.0, 0.5 -> Misaligned -> No Decay
    # Index 4: 0.0, 1.0 -> Zero param -> Decay (0 * s = 0) effectively
    # no change but formula applies.
    # Note: 0.0 * 1.0 >= 0 is True. So decay is applied:
    # u + s * p = 1.0 + s * 0.0 = 1.0.

    decay_rate = 0.1
    transform = _cwd.add_cautious_weight_decay(weight_decay=decay_rate)
    state = transform.init(params)
    updates_out, _ = transform.update(updates, state, params)

    expected_updates = {
        'x': jnp.array([
            0.5 + 0.1 * 1.0,   # Aligned
            -0.5 + 0.1 * -1.0,  # Aligned
            -0.5,              # Misaligned
            0.5,               # Misaligned
            1.0 + 0.1 * 0.0    # Aligned (0 >= 0)
        ])
    }

    test_utils.assert_trees_all_close(updates_out, expected_updates)

  def test_cautious_weight_decay_mask(self):
    params = {'x': jnp.array(1.0), 'y': jnp.array(1.0)}
    updates = {'x': jnp.array(1.0), 'y': jnp.array(1.0)}
    # Both aligned.

    decay_rate = 0.1
    # Only decay 'x', not 'y'
    mask = {'x': True, 'y': False}

    transform = _cwd.add_cautious_weight_decay(
        weight_decay=decay_rate, mask=mask
    )
    state = transform.init(params)
    updates_out, _ = transform.update(updates, state, params)

    expected_updates = {
        'x': jnp.array(1.0 + 0.1 * 1.0),  # Masked=True, Aligned -> Decay
        'y': jnp.array(1.0),  # Masked=False -> No Decay
    }

    test_utils.assert_trees_all_close(updates_out, expected_updates)

  def test_cautious_weight_decay_schedule(self):
    params = {'x': jnp.array([1.0])}
    updates = {'x': jnp.array([1.0])}
    # Aligned

    def schedule(count):
      # 0.1 at step 0, 0.2 at step 1
      return 0.1 * (count + 1)

    transform = _cwd.add_cautious_weight_decay(weight_decay=schedule)
    state = transform.init(params)

    # Step 0
    updates_out, state = transform.update(updates, state, params)
    expected_step0 = {'x': jnp.array([1.0 + 0.1 * 1.0])}
    test_utils.assert_trees_all_close(updates_out, expected_step0)

    # Step 1
    updates_out, state = transform.update(updates, state, params)
    expected_step1 = {'x': jnp.array([1.0 + 0.2 * 1.0])}
    test_utils.assert_trees_all_close(updates_out, expected_step1)


if __name__ == '__main__':
  absltest.main()
