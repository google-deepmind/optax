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
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import test_utils
from optax._src import update
from optax.transforms import _freezing

# Data setup for Freeze shortcuts test

PARAMS_FLAT = {"a": np.zeros(1), "b": np.ones(2)}
PARAMS_NESTED = {
    "layer1": [np.zeros((2, 3)), {"bias": np.ones(3)}],
    "layer2": {"w": np.full((4, 2), 2.0), "b": np.full(4, -1.0)},
    "misc": (np.array(5.0),),
}
GRAD_FLAT = {"a": np.array([1.0]), "b": np.array([2.0, 3.0])}
GRAD_NESTED = {
    "layer1": [np.full((2, 3), 1.0), {"bias": np.full(3, 2.0)}],
    "layer2": {"w": np.full((4, 2), 3.0), "b": np.full(4, 4.0)},
    "misc": (np.array(1.0),),
}


class FreezeTest(parameterized.TestCase):

  @parameterized.named_parameters([
      # flat: freeze only 'a'
      (
          "flat_freeze_a",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"a": True, "b": False},
          {"a": np.array([0.0]), "b": GRAD_FLAT["b"]},
      ),
      # flat: freeze only 'b'
      (
          "flat_freeze_b",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"a": False, "b": True},
          {"a": GRAD_FLAT["a"], "b": np.array([0.0, 0.0])},
      ),
      # flat: freeze everything
      (
          "flat_freeze_all",
          PARAMS_FLAT,
          GRAD_FLAT,
          True,
          {"a": np.array([0.0]), "b": np.array([0.0, 0.0])},
      ),
      # flat: freeze nothing
      ("flat_freeze_none", PARAMS_FLAT, GRAD_FLAT, False, GRAD_FLAT),
      # nested: freeze first layer1 weight only
      (
          "nested_freeze_l1_0",
          PARAMS_NESTED,
          GRAD_NESTED,
          {
              "layer1": [True, {"bias": False}],
              "layer2": {"w": False, "b": False},
              "misc": (False,),
          },
          {
              "layer1": [
                  np.zeros_like(GRAD_NESTED["layer1"][0]),
                  GRAD_NESTED["layer1"][1],
              ],
              "layer2": GRAD_NESTED["layer2"],
              "misc": GRAD_NESTED["misc"],
          },
      ),
      # nested: freeze only layer2['w']
      (
          "nested_freeze_l2_w",
          PARAMS_NESTED,
          GRAD_NESTED,
          {
              "layer1": [False, {"bias": False}],
              "layer2": {"w": True, "b": False},
              "misc": (False,),
          },
          {
              "layer1": GRAD_NESTED["layer1"],
              "layer2": {
                  "w": np.zeros_like(GRAD_NESTED["layer2"]["w"]),
                  "b": GRAD_NESTED["layer2"]["b"],
              },
              "misc": GRAD_NESTED["misc"],
          },
      ),
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

  def test_nested_freeze_all(self):
    mask = {
        "layer1": [True, {"bias": True}],
        "layer2": {"w": True, "b": True},
        "misc": (True,),
    }
    opt = _freezing.freeze(mask)
    state = opt.init(PARAMS_NESTED)
    updates, _ = opt.update(GRAD_NESTED, state, PARAMS_NESTED)
    test_utils.assert_trees_all_close(
        updates, jax.tree.map(lambda g: g * 0, GRAD_NESTED), atol=0
    )

  def test_nested_freeze_none(self):
    mask = {
        "layer1": [False, {"bias": False}],
        "layer2": {"w": False, "b": False},
        "misc": (False,),
    }
    opt = _freezing.freeze(mask)
    state = opt.init(PARAMS_NESTED)
    updates, _ = opt.update(GRAD_NESTED, state, PARAMS_NESTED)
    test_utils.assert_trees_all_close(updates, GRAD_NESTED, atol=0)

  @parameterized.named_parameters([
      ("py_bool", True),
      ("jax_bool", np.array(False)),
  ])
  def test_scalar_bool_broadcast(self, scalar_mask):
    opt = _freezing.freeze(scalar_mask)
    state = opt.init(PARAMS_FLAT)
    updates, _ = opt.update(GRAD_FLAT, state, PARAMS_FLAT)
    expected = (
        jax.tree.map(jnp.zeros_like, PARAMS_FLAT)
        if bool(scalar_mask)
        else GRAD_FLAT
    )
    test_utils.assert_trees_all_close(updates, expected, atol=0)

  def test_bad_structure_raises(self):
    bad_mask = {"layer1": [True]}  # missing the bias leaf
    opt = _freezing.freeze(bad_mask)
    with self.assertRaisesRegex(ValueError, "Dict key mismatch"):
      opt.update(GRAD_NESTED, opt.init(PARAMS_NESTED), PARAMS_NESTED)

  def test_partial_prefix_mask_behavior(self):
    """Tests freeze behavior with masks that are prefixes."""
    params = {"a": 1.0, "b": {"c": 2.0, "d": 3.0}}
    grads = {"a": 10.0, "b": {"c": 20.0, "d": 30.0}}
    grads = jax.tree.map(jnp.asarray, grads)
    # Mask is a prefix: True for 'b' applies to the whole subtree {'c', 'd'}.

    mask = {"a": False, "b": True}
    optimizer = _freezing.freeze(mask)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)

    # Expect 'a' grads to pass through, 'b' subtree grads to be zeroed.

    expected_updates = {
        "a": jnp.array(10.0),
        "b": {"c": jnp.array(0.0), "d": jnp.array(0.0)},
    }
    test_utils.assert_trees_all_close(updates, expected_updates, atol=0)


class SelectiveTransformTest(parameterized.TestCase):

  @parameterized.named_parameters([
      # flat: freeze b only
      (
          "flat_freeze_b",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"a": False, "b": True},
          {
              "a": PARAMS_FLAT["a"] - GRAD_FLAT["a"],
              "b": PARAMS_FLAT["b"],
          },
      ),
      # flat: freeze a only
      (
          "flat_freeze_a",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"a": True, "b": False},
          {
              "a": PARAMS_FLAT["a"],
              "b": PARAMS_FLAT["b"] - GRAD_FLAT["b"],
          },
      ),
      # flat: freeze none (explicit full mask)
      (
          "flat_freeze_none",
          PARAMS_FLAT,
          GRAD_FLAT,
          {"a": False, "b": False},
          jax.tree.map(lambda p, g: p - g, PARAMS_FLAT, GRAD_FLAT),
      ),
      # flat: freeze all (scalar)
      ("flat_freeze_all", PARAMS_FLAT, GRAD_FLAT, True, PARAMS_FLAT),
      # nested: freeze layer1 weights only
      (
          "nested_freeze_layer1_weight",
          PARAMS_NESTED,
          GRAD_NESTED,
          {
              "layer1": [True, {"bias": False}],
              "layer2": {"w": False, "b": False},
              "misc": (False,),
          },
          {
              "layer1": [
                  PARAMS_NESTED["layer1"][0],  # frozen
                  {
                      "bias": (
                          PARAMS_NESTED["layer1"][1]["bias"]
                          - GRAD_NESTED["layer1"][1]["bias"]
                      )
                  },
              ],
              "layer2": {
                  "w": (
                      PARAMS_NESTED["layer2"]["w"] - GRAD_NESTED["layer2"]["w"]
                  ),
                  "b": (
                      PARAMS_NESTED["layer2"]["b"] - GRAD_NESTED["layer2"]["b"]
                  ),
              },
              "misc": (PARAMS_NESTED["misc"][0] - GRAD_NESTED["misc"][0],),
          },
      ),
      # nested: freeze entire layer2
      (
          "nested_freeze_layer2",
          PARAMS_NESTED,
          GRAD_NESTED,
          {
              "layer1": [False, {"bias": False}],
              "layer2": {"w": True, "b": True},
              "misc": (False,),
          },
          {
              "layer1": [
                  PARAMS_NESTED["layer1"][0] - GRAD_NESTED["layer1"][0],
                  {
                      "bias": (
                          PARAMS_NESTED["layer1"][1]["bias"]
                          - GRAD_NESTED["layer1"][1]["bias"]
                      )
                  },
              ],
              "layer2": PARAMS_NESTED["layer2"],  # frozen
              "misc": (PARAMS_NESTED["misc"][0] - GRAD_NESTED["misc"][0],),
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

  def test_nested_train_all(self):
    mask = {
        "layer1": [False, {"bias": False}],
        "layer2": {"w": False, "b": False},
        "misc": (False,),
    }
    opt = _freezing.selective_transform(alias.sgd(1.0), freeze_mask=mask)
    updates, _ = opt.update(GRAD_NESTED, opt.init(PARAMS_NESTED), PARAMS_NESTED)
    new_params = update.apply_updates(PARAMS_NESTED, updates)
    expected = jax.tree.map(lambda p, g: p - g, PARAMS_NESTED, GRAD_NESTED)
    test_utils.assert_trees_all_close(new_params, expected, atol=1e-6)

  @parameterized.named_parameters([
      ("py_bool", True),
      ("jax_bool", np.array(True)),
  ])
  def test_scalar_freeze_all(self, scalar_mask):
    opt = _freezing.selective_transform(alias.sgd(1.0), freeze_mask=scalar_mask)
    updates, _ = opt.update(GRAD_FLAT, opt.init(PARAMS_FLAT), PARAMS_FLAT)
    new_params = update.apply_updates(PARAMS_FLAT, updates)
    test_utils.assert_trees_all_close(new_params, PARAMS_FLAT, atol=1e-6)

  def test_selective_bad_structure(self):
    bad_mask = {"a": True}  # missing 'b'
    opt = _freezing.selective_transform(alias.sgd(1.0), freeze_mask=bad_mask)
    with self.assertRaisesRegex(ValueError, "Dict key mismatch"):
      opt.update(GRAD_FLAT, opt.init(PARAMS_FLAT), PARAMS_FLAT)

  def test_partial_prefix_mask_behavior(self):
    """Tests selective_transform behavior with masks that are prefixes."""
    params = {"a": 1.0, "b": {"c": 2.0, "d": 3.0}}
    grads = {"a": 10.0, "b": {"c": 20.0, "d": 30.0}}
    params = jax.tree.map(jnp.asarray, params)
    grads = jax.tree.map(jnp.asarray, grads)

    # Mask is a prefix: True for 'b' means freeze the whole subtree {'c', 'd'}.

    mask = {"a": False, "b": True}
    inner_opt = alias.sgd(learning_rate=1.0)  # SGD subtracts the gradient
    optimizer = _freezing.selective_transform(inner_opt, freeze_mask=mask)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    new_params = update.apply_updates(params, updates)

    # 'a' is updated by SGD (p - g), 'b' remains unchanged

    expected_params = {"a": params["a"] - grads["a"], "b": params["b"]}
    test_utils.assert_trees_all_close(new_params, expected_params, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
