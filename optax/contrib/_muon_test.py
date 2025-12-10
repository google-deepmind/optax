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
from optax._src import numerics
from optax._src import test_utils
from optax._src import update
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


def _setup_mixed_tensor_target_complex(dtype):
  """Complex version of _common_test._setup_mixed_tensor_target."""
  initial_params = jnp.zeros((2, 2), dtype=dtype)
  final_params = jnp.array(
      [[1.0+2.0j, 0.0], [-1.0+1.0j, 1.0-3.0j]],
      dtype=dtype,
  )

  def obj_fn(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, obj_fn


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

  def test_newton_schulz(self):
    """Test that Newton--Schulz orhogonalizes/unitiarizes correctly."""
    mat_real = jax.random.normal(jax.random.key(0), (4, 3), dtype=jnp.float32)
    mat_complex = jax.random.normal(
        jax.random.key(0), (4, 3), dtype=jnp.complex64
    )

    ns_coeffs = jnp.array([2.0, -1.5, 0.5])

    if jax.default_backend() == "tpu":
      atol, rtol = 1e-2, 1e-2
    else:
      atol, rtol = 1e-5, 1e-5

    # For real matrices, Newton--Schulz should produce an orthonormal matrix
    mat_real_orth = _muon.orthogonalize_via_newton_schulz(
        mat_real,
        ns_coeffs,
        ns_steps=20,
        eps=1e-12,
        dimension_numbers=_muon.MuonDimensionNumbers(0, 1),
    )

    gram_real = mat_real_orth.T @ mat_real_orth
    with self.subTest("Real Newton--Schulz produces an orthonormal matrix"):
      self.assertTrue(
          jnp.allclose(
              gram_real, jnp.eye(mat_real.shape[1]), atol=atol, rtol=rtol
          ),
          msg=(
              "Real Newton–Schulz did not orthogonalize correctly."
              f"\nGram:\n{gram_real}"
          ),  # should be close to identity
      )

    # For complex matrices, Newton--Schulz should produce a unitary matrix
    mat_complex_orth = _muon.orthogonalize_via_newton_schulz(
        mat_complex,
        ns_coeffs,
        ns_steps=10,
        eps=1e-8,
        dimension_numbers=_muon.MuonDimensionNumbers(0, 1),
    )

    gram_complex = mat_complex_orth.conj().T @ mat_complex_orth
    with self.subTest("Complex Newton--Schulz produces a unitary matrix"):
      self.assertTrue(
          jnp.allclose(
              gram_complex, jnp.eye(mat_complex.shape[1]), atol=atol, rtol=rtol
          ),
          msg=(
              "Complex Newton–Schulz did not produce a unitary matrix."
              f"\nGram:\n{gram_complex}"
          ),  # should be close to identity
      )

    with self.subTest("Output shape is preserved for complex matrices"):
      # Check that the output shape is preserved.
      self.assertEqual(mat_complex_orth.shape, mat_complex.shape)

  @parameterized.product(
      target=(_setup_mixed_tensor_target_complex,),
      dtype=("complex64",),
      adaptive=(True, False),
      nesterov=(True, False),
  )
  def test_complex_mixed_target(self, target, dtype, adaptive, nesterov):
    """Test Muon optimizer on a complex mixed tensor optimization target."""
    dtype = getattr(jnp, dtype)

    opt = _muon.muon(
        learning_rate=1e-2,
        adaptive=adaptive,
        nesterov=nesterov,
    )
    initial_params, final_params, obj_fn = target(dtype)

    @jax.jit
    def step(params, state):
      _, updates = jax.value_and_grad(obj_fn)(params)
      updates = jax.tree.map(jnp.conj, updates)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)

    def f(params_state, _):
      return step(*params_state), None

    (params, _), _ = jax.lax.scan(f, (params, state), length=1000)

    test_utils.assert_trees_all_close(
        params, final_params, rtol=3e-2, atol=3e-2
    )

if __name__ == "__main__":
  absltest.main()
