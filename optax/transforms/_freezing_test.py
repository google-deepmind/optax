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

"""Tests for methods in `optax.transforms._freezing.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import test_utils
from optax._src import update
from optax.transforms import _freezing


PARAMS_FLAT = {"w": np.array([1.0, 2.0]), "b": np.array([3.0])}
GRAD_FLAT = {"w": np.array([0.1, 0.2]), "b": np.array([0.3])}

PARAMS_NESTED = {
    "layer1": {"w": np.array([[1.0, 2.0]]), "b": np.array([3.0])},
    "layer2": {"w": np.array([[4.0, 5.0]])},
}
GRAD_NESTED = {
    "layer1": {"w": np.array([[0.1, 0.2]]), "b": np.array([0.3])},
    "layer2": {"w": np.array([[0.4, 0.5]])},
}


class FreezeTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          "flat_partial",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"w": True, "b": False},
          {"w": np.array([0.0, 0.0]), "b": np.array([0.3])},
      ),
      (
          "nested_partial",
          PARAMS_NESTED,
          GRAD_NESTED,
          {"layer1": {"w": True, "b": False}, "layer2": {"w": False}},
          {
              "layer1": {
                  "w": np.array([[0.0, 0.0]]),
                  "b": np.array([0.3]),
              },
              "layer2": {"w": np.array([[0.4, 0.5]])},
          },
      ),
      (
          "freeze_all",
          PARAMS_NESTED,
          GRAD_NESTED,
          True,
          {
              "layer1": {
                  "w": np.array([[0.0, 0.0]]),
                  "b": np.array([0.0]),
              },
              "layer2": {"w": np.array([[0.0, 0.0]])},
          },
      ),
      ("freeze_none", PARAMS_NESTED, GRAD_NESTED, False, GRAD_NESTED),
  ])
  def test_freeze_updates(self, params, grads, freeze_mask, expected_updates):
    """Tests that freeze zeros out the correct gradient updates."""
    optimizer = _freezing.freeze(freeze_mask)
    state = optimizer.init(params)
    updates, new_state = optimizer.update(grads, state, params)

    test_utils.assert_trees_all_close(updates, expected_updates, atol=0)
    test_utils.assert_trees_all_equal(state, new_state)
    # pytype: disable=attribute-error
    test_utils.assert_trees_all_equal(state.inner_state, base.EmptyState())
    # pytype: enable=attribute-error

  def test_bad_structure_raises(self):
    bad_mask = {"layer1": {"w": True}}  # missing 'b'
    opt = _freezing.freeze(bad_mask)
    with self.assertRaises(ValueError):
      opt.update(GRAD_NESTED, opt.init(PARAMS_NESTED), PARAMS_NESTED)


class SelectiveTransformTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          "flat_selective",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"w": True, "b": False},
          {"w": np.array([1.0, 2.0]), "b": np.array([2.7])},
      ),
      (
          "nested_selective",
          PARAMS_NESTED,
          GRAD_NESTED,
          {"layer1": {"w": True, "b": False}, "layer2": {"w": False}},
          {
              "layer1": {
                  "w": np.array([[1.0, 2.0]]),
                  "b": np.array([2.7]),
              },
              "layer2": {"w": np.array([[3.6, 4.5]])},
          },
      ),
  ])
  def test_selective_transform_effect(
      self, params, grads, mask, expected_params
  ):
    """Tests that selective_transform only updates the unfrozen leaves."""
    inner_opt = alias.sgd(learning_rate=1.0)
    optimizer = _freezing.selective_transform(inner_opt, freeze_mask=mask)

    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    new_params = update.apply_updates(params, updates)

    test_utils.assert_trees_all_close(new_params, expected_params, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
