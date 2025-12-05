# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for DoG optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import warnings
from optax._src import combine
from optax.contrib import _dog as dog
from optax._src import test_utils
from optax._src import transform
from optax._src import update
import optax.tree


class DoGTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  @parameterized.named_parameters([
      ('dog', dog.scale_by_dog),
      ('dowg', dog.scale_by_dowg),
      ('l_dog', dog.scale_by_l_dog),
  ])
  def test_scalers(self, scaler_constr):
    params = self.init_params
    if scaler_constr is dog.scale_by_dog:
        scaler = scaler_constr(init_step=("heuristic", 1e-6))
    elif scaler_constr is dog.scale_by_l_dog:
        scaler = scaler_constr(reps_rel=1e-6)
    else:
        scaler = scaler_constr()

    init_fn = jax.jit(scaler.init)
    transform_fn = jax.jit(scaler.update)

    state = init_fn(params)
    test_utils.assert_tree_all_finite(state)

    updates, state = transform_fn(
        self.per_step_updates, state, params
    )
    test_utils.assert_tree_all_finite((params, updates, state))
    test_utils.assert_trees_all_equal_shapes(params, updates)

  def test_l_dog_vs_dog(self):
    # Test that scale_by_l_dog produces different results than scale_by_dog
    # and that it works.
    params = self.init_params

    # Global DoG
    scaler_global = dog.scale_by_dog(
        init_step=("heuristic", 1e-6)
    )
    state_global = scaler_global.init(params)
    updates_global, _ = scaler_global.update(
        self.per_step_updates, state_global, params
    )

    # Layer-wise DoG
    scaler_layer = dog.scale_by_l_dog(reps_rel=1e-6)
    state_layer = scaler_layer.init(params)
    updates_layer, _ = scaler_layer.update(
        self.per_step_updates, state_layer, params
    )


    with self.assertRaises(AssertionError):
        test_utils.assert_trees_all_close(updates_global, updates_layer)

  def test_legacy_compatibility(self):
    # Test that scale_by_distance_over_gradients matches scale_by_l_dog

    params = self.init_params
    reps_rel = 1e-6
    global_scale = 1.0

    # Legacy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy_scaler = transform.scale_by_distance_over_gradients(
            reps_rel=reps_rel, global_scale=global_scale
        )
    legacy_state = legacy_scaler.init(params)
    legacy_updates, _ = legacy_scaler.update(
        self.per_step_updates, legacy_state, params
    )

    # New - using scale_by_l_dog directly
    l_dog_scaler = combine.chain(
        dog.scale_by_l_dog(reps_rel=reps_rel),
        transform.scale(global_scale),
    )
    l_dog_state = l_dog_scaler.init(params)
    l_dog_updates, _ = l_dog_scaler.update(
        self.per_step_updates, l_dog_state, params
    )

    test_utils.assert_trees_all_close(legacy_updates, l_dog_updates)


if __name__ == '__main__':
  absltest.main()
