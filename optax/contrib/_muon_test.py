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
"""Tests for the Muon optimizer in `muon.py`."""


import math
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax.contrib import _muon
from optax.transforms import _masking

UNSPECIFIED = object()


def get_updates(params, muon_weight_dimension_numbers=UNSPECIFIED):
  if muon_weight_dimension_numbers is UNSPECIFIED:
    opt = _muon.muon(learning_rate=1e-3)
  else:
    opt = _muon.muon(
        learning_rate=1e-3,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers
    )
  state = opt.init(params)
  # assume loss = 1/2 * sum(params ** 2)
  grad = params
  updates, state = opt.update(grad, state, params=params)
  return updates, state


class MuonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "2d_tuple_axes",
          "input_shape": (2, 3),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=0,
                                                 output_axis=1),
          "expected_flat_shape": (1, 2, 3),
      },
      {
          "testcase_name": "3d_batch_axis",
          "input_shape": (2, 3, 4),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=0,
                                                 output_axis=2),
          "expected_flat_shape": (3, 2, 4),
      },
      {
          "testcase_name": "3d_negative_axes_indices",
          "input_shape": (2, 3, 4),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=-3,
                                                 output_axis=-1),
          "expected_flat_shape": (3, 2, 4),
      },
      {
          "testcase_name": "4d_multiple_batch_axes",
          "input_shape": (2, 3, 4, 5),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=2,
                                                 output_axis=3),
          "expected_flat_shape": (6, 4, 5),
      },
      {
          "testcase_name": "4d_multiple_output_axes",
          "input_shape": (2, 3, 4, 5),
          "dim_nums": _muon.MuonDimensionNumbers(reduction_axis=2,
                                                 output_axis=(0, 3)),
          "expected_flat_shape": (3, 4, 10),
      },
  )
  def test_reshape_inverse(self, input_shape, dim_nums, expected_flat_shape):
    x = jnp.arange(math.prod(input_shape), dtype=jnp.float32).reshape(
        input_shape
    )
    reshape_fn, inverse_fn = _muon._compute_muon_reshape(x, dim_nums)
    reshaped_x = reshape_fn(x)
    reconstructed_x = inverse_fn(reshaped_x)
    # Check flat shape (batch, reduction, output)
    self.assertEqual(reshaped_x.shape, expected_flat_shape)
    # Check inverse shape and value
    self.assertEqual(reconstructed_x.shape, x.shape)
    test_utils.assert_trees_all_close(reconstructed_x, x)

  def test_callable_weight_dim_nums(self):
    # Case 1: a dim nums for all weights, no matter if they're muon.
    def weight_dim_nums_fn(params):
      fn_ = lambda x: _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None
      return jax.tree.map(fn_, params)

    opt = _muon.muon(learning_rate=1e-3,
                     muon_weight_dimension_numbers=weight_dim_nums_fn)
    params = {"w1": jnp.ones((10, 10)), "w2": jnp.ones((2, 10))}
    state = opt.init(params)
    _, _ = opt.update(params, state, params=params)

    # Case 2: a None inserted for parameters that are not muon.
    def weight_dim_nums_fn(params):  # pylint: disable=function-redefined
      del params
      return {"w1": _muon.MuonDimensionNumbers(), "w2": None}

    opt = _muon.muon(learning_rate=1e-3,
                     muon_weight_dimension_numbers=weight_dim_nums_fn)
    params = {"w1": jnp.ones((10, 10)), "w2": jnp.ones((2, 10))}
    state = opt.init(params)
    _, _ = opt.update(params, state, params=params)

  def test_reshape_update_for_square_parameter_matches_muon_without_dim_nums(
      self
  ):
    # Use 2D parameter (10, 10) with no dim nums as groundtruth
    key = jax.random.key(0)
    params_sq = {"w": jax.random.normal(key, shape=(10, 10))}
    updates_sq, _ = get_updates(params_sq)
    # Test: 2D parameter (10, 10) with trivial dim nums
    dim_nums = {
        "w": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1)}
    reshape_updates_sq, _ = get_updates(params_sq,
                                        muon_weight_dimension_numbers=dim_nums)
    test_utils.assert_trees_all_close(
        updates_sq, reshape_updates_sq, rtol=1e-8, atol=1e-8
    )

  def test_reshape_and_update_single_param(self):
    # Use 2D parameter (10, 12) with no dimension numbers as groundtruth
    key = jax.random.key(0)
    w = jax.random.normal(key, shape=(10, 12))
    params = {"w": w}
    updates, _ = get_updates(params)

    with self.subTest("2D with dimension numbers, (10, 12)"):
      # Test 1: 2D with dimension numbers, (10, 12)
      params = {"w": w}
      dim_nums = {
          "w": _muon.MuonDimensionNumbers(reduction_axis=0, output_axis=1)}
      reshape_updates, _ = get_updates(params,
                                       muon_weight_dimension_numbers=dim_nums)
      test_utils.assert_trees_all_close(updates, reshape_updates, rtol=1e-8,
                                        atol=1e-8)

    with self.subTest("4D with dim nums, (10, 12) -> (4, 1, 10, 3)"):
      # Test 2: 4D with dim nums, (10, 12) -> (4, 1, 10, 3)
      reshape_fn = lambda x: x.reshape(10, 3, 1, 4).transpose(3, 2, 0, 1)
      reshape_params = {"w": reshape_fn(w)}
      dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=(2,),
                                                  output_axis=(0, 3))}
      reshape_updates, _ = get_updates(reshape_params,
                                       muon_weight_dimension_numbers=dim_nums)
      test_utils.assert_trees_all_close(
          jax.tree.map(reshape_fn, updates), reshape_updates, rtol=1e-8,
          atol=1e-8)

    with self.subTest("4D with dim_nums, (10, 12) -> (5, 12, 1, 2)"):
      # Test 3: 4D with dim_nums, (10, 12) -> (5, 12, 1, 2)
      reshape_fn = lambda x: x.reshape(2, 1, 5, 12).transpose(2, 3, 1, 0)
      reshape_params = {"w": reshape_fn(w)}
      dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=(0, 3),
                                                  output_axis=(1,))}
      reshape_updates, _ = get_updates(reshape_params,
                                       muon_weight_dimension_numbers=dim_nums)
      test_utils.assert_trees_all_close(
          jax.tree.map(reshape_fn, updates), reshape_updates, rtol=1e-8,
          atol=1e-8)

  def test_dim_nums_combinations(self):
    get_muon_mu = lambda state: state[0]["muon"][0][0][1]
    dim_num = _muon.MuonDimensionNumbers(reduction_axis=(1,),
                                         output_axis=(2,))

    # Test 1: full dim_nums
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((2, 3, 4))}
    dim_nums = {"w1": dim_num, "w2": dim_num}
    _, state = get_updates(params, dim_nums)
    self.assertNotIsInstance(get_muon_mu(state)["w1"], _masking.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"], _masking.MaskedNode)

    # Test 2: no dim_nums
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((3, 4))}
    _, state = get_updates(params)
    self.assertIsInstance(get_muon_mu(state)["w1"], _masking.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"], _masking.MaskedNode)

    # Test 3: partial dim_nums with none
    params = {"w1": jnp.ones((1, 2, 3)), "w2": jnp.ones((2, 3, 4))}
    dim_nums = {"w1": None, "w2": dim_num}
    _, state = get_updates(params, dim_nums)
    self.assertIsInstance(get_muon_mu(state)["w1"], _masking.MaskedNode)

    # Test 4: prefix None, full dim_nums
    params = {
        "w1": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
        "w2": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
    }
    dim_num = _muon.MuonDimensionNumbers()
    dim_nums = {"w1": None, "w2": {"a": dim_num, "b": None}}
    _, state = get_updates(params, dim_nums)
    state_structure = jax.tree.structure(
        get_muon_mu(state),
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode))
    self.assertEqual(state_structure, jax.tree.structure(params))
    self.assertIsInstance(get_muon_mu(state)["w1"]["a"], _masking.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w1"]["b"], _masking.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w2"]["a"], _masking.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["b"], _masking.MaskedNode)

    # Test 5: prefix None and dim_nums
    params = {
        "w1": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
        "w2": {"a": jnp.ones((2, 3)), "b": jnp.ones((2, 3))},
    }
    dim_num = _muon.MuonDimensionNumbers()
    dim_nums = {"w1": dim_num, "w2": None}
    _, state = get_updates(params, dim_nums)
    state_structure = jax.tree.structure(
        get_muon_mu(state),
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode))
    self.assertEqual(state_structure, jax.tree.structure(params))
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["a"], _masking.MaskedNode)
    self.assertNotIsInstance(get_muon_mu(state)["w1"]["b"], _masking.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["a"], _masking.MaskedNode)
    self.assertIsInstance(get_muon_mu(state)["w2"]["b"], _masking.MaskedNode)


if __name__ == "__main__":
  absltest.main()
