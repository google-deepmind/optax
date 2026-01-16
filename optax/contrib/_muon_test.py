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
import optax


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
      [[1.0 + 2.0j, 0.0], [-1.0 + 1.0j, 1.0 - 3.0j]],
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

  def test_muon_adamw_lr_default(self):
    """Test that if learning_rate_adam is None, behavior is identical to
    current."""
    # Use simple params: 2D for Muon, 1D for AdamW
    params = {"w": jnp.ones((5, 5)), "b": jnp.ones(5)}
    grads = {"w": jnp.ones((5, 5)) * 0.1, "b": jnp.ones(5) * 0.1}

    # Case 1: No learning_rate_adam (default)
    opt_default = _muon.muon(learning_rate=0.01)
    state_default = opt_default.init(params)
    updates_default, _ = opt_default.update(grads, state_default, params)

    # Case 2: learning_rate_adam explicitly set to same value
    opt_explicit = _muon.muon(learning_rate=0.01, learning_rate_adam=0.01)
    state_explicit = opt_explicit.init(params)
    updates_explicit, _ = opt_explicit.update(grads, state_explicit, params)

    # Verify exact numerical identity
    test_utils.assert_trees_all_close(
        updates_default, updates_explicit, rtol=0.0, atol=0.0)

    # Case 3: learning_rate_adam set to None
    opt_none = _muon.muon(learning_rate=0.01, learning_rate_adam=None)
    state_none = opt_none.init(params)
    updates_none, _ = opt_none.update(grads, state_none, params)

    # Verify exact numerical identity
    test_utils.assert_trees_all_close(
        updates_default, updates_none, rtol=0.0, atol=0.0)

  def test_muon_separate_adamw_learning_rate(self):
    """Test that separate learning rates lead to different updates."""
    params = {
        "w": jnp.ones((10, 10)),    # 2D - uses Muon
        "b": jnp.ones(10),          # 1D - uses AdamW
        "ln_w": jnp.ones((5, 5)),   # 2D - uses Muon
    }
    grads = {
        "w": jnp.ones((10, 10)) * 0.1,
        "b": jnp.ones(10) * 0.1,
        "ln_w": jnp.ones((5, 5)) * 0.1,
    }

    # Optimizer with different learning rates
    # Muon LR = 0.01, AdamW LR = 0.001
    opt = _muon.muon(learning_rate=0.01, learning_rate_adam=0.001)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    # Reference optimizer with global LR = 0.01
    opt_ref = _muon.muon(learning_rate=0.01)
    state_ref = opt_ref.init(params)
    updates_ref, _ = opt_ref.update(grads, state_ref, params)

    # 2D tensors should be identical (both use LR=0.01 via Muon)
    test_utils.assert_trees_all_close(updates["w"], updates_ref["w"])
    test_utils.assert_trees_all_close(updates["ln_w"], updates_ref["ln_w"])

    # 1D tensors should be different (AdamW uses 0.001 vs 0.01)
    # Since updates scale linearly with LR, expect significant difference
    self.assertFalse(jnp.allclose(updates["b"], updates_ref["b"]))

    # Additional check: Create optimizer with LR=0.001 globally
    opt_low = _muon.muon(learning_rate=0.001)
    state_low = opt_low.init(params)
    updates_low, _ = opt_low.update(grads, state_low, params)

    # 1D tensors should match the global LR=0.001 case
    test_utils.assert_trees_all_close(updates["b"], updates_low["b"])

  def test_muon_adamw_lr_extreme_differences(self):
    """Test with extreme differences in learning rates (100x ratio)."""
    # Paper shows ratios like Muon=0.1, Adam=0.001 are useful
    params = {"w": jnp.ones((5, 5)), "b": jnp.ones(5)}
    # Use eye for w to ensure full rank for Newton-Schulz
    grads = {"w": jnp.eye(5) * 0.01, "b": jnp.ones(5) * 0.01}

    opt = _muon.muon(learning_rate=0.1, learning_rate_adam=0.001)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    # Ensure no NaNs or Infs
    self.assertTrue(jnp.all(jnp.isfinite(updates["w"])))
    self.assertTrue(jnp.all(jnp.isfinite(updates["b"])))

    # Check that magnitudes correspond roughly to LRs
    # Muon update is normalized, scaled by LR=0.1. Norm approx 0.1
    # Adam update is normalized, scaled by LR=0.001. Norm approx 0.001
    # b_norm should be much smaller than w_norm (roughly 1/100th)
    # Since w is (5,5) and b is (5), let's look at mean abs value
    w_mean = jnp.mean(jnp.abs(updates["w"]))
    b_mean = jnp.mean(jnp.abs(updates["b"]))

    # Should differ by order of magnitude
    self.assertGreater(w_mean, 10 * b_mean)

  def test_muon_adamw_lr_param_grouping(self):
    """Verify correct parameters go to Muon vs Adam based on
    dimension_numbers."""
    params = {
        "muon_param": jnp.ones((10, 10)),      # 2D, default Muon
        "adam_param": jnp.ones(10),            # 1D, default Adam
        "force_adam": jnp.ones((5, 5)),        # 2D, but we will force to Adam
    }
    grads = jax.tree.map(lambda x: x * 0.1, params)

    # Force "force_adam" to be Adam by using mask
    def dim_nums_fn(params):
        return {
            "muon_param": _muon.MuonDimensionNumbers(0, 1),
            "adam_param": None,
            "force_adam": None  # Explicitly force this 2D param to Adam
        }

    # Set vastly different LRs to detect which optimizer is used
    muon_lr = 1.0
    adam_lr = 0.0001

    opt = _muon.muon(
        learning_rate=muon_lr,
        learning_rate_adam=adam_lr,
        muon_weight_dimension_numbers=dim_nums_fn
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    # Check magnitudes
    # muon_param should obey muon_lr (large update)
    # Frobenius norm=1.0, RMS=1/sqrt(N). For 10x10, RMS=0.1.

    # We expect muon update to be significantly larger than adam update
    muon_update_mean = jnp.mean(jnp.abs(updates["muon_param"]))
    adam_update_mean = jnp.mean(jnp.abs(updates["force_adam"]))

    # Assert muon is at least 100x larger (0.1 vs 0.0001)
    self.assertGreater(muon_update_mean, 100 * adam_update_mean)

    # Also rudimentary absolute check
    self.assertGreater(muon_update_mean, 0.02)
    self.assertLess(adam_update_mean, 0.01)

  def test_muon_adamw_lr_schedule_compatibility(self):
    """Test that two different schedule objects work correctly."""
    # Schedule 1: Constant
    schedule_main = optax.constant_schedule(0.01)
    # Schedule 2: Exponential
    schedule_adam = optax.exponential_decay(0.001, 100, 0.5)

    opt = _muon.muon(
        learning_rate=schedule_main,
        learning_rate_adam=schedule_adam
    )

    params = {"w": jnp.ones((5, 5)), "b": jnp.ones(5)}
    grads = {"w": jnp.ones((5, 5)) * 0.1, "b": jnp.ones(5) * 0.1}
    state = opt.init(params)
    # Should run without error
    updates, _ = opt.update(grads, state, params)
    self.assertTrue(jnp.all(jnp.isfinite(updates["w"])))

  def test_muon_adamw_lr_schedule_evolution(self):
    """Verify schedules evolve independently."""

    # Main: Decay fast
    schedule_main = optax.exponential_decay(1.0, 1, 0.5)
    # Adam: Decay slow
    schedule_adam = optax.exponential_decay(1.0, 1, 0.9)

    opt = _muon.muon(
        learning_rate=schedule_main,
        learning_rate_adam=schedule_adam
    )

    params = {"w": jnp.ones((5, 5)), "b": jnp.ones(5)}
    grads = {"w": jnp.ones((5, 5)), "b": jnp.ones(5)}  # Unit gradients
    state = opt.init(params)
    # Step 0: Both LRs = 1.0 (approx)
    updates0, state = opt.update(grads, state, params)

    # Step 1: Main=0.5, Adam=0.9
    _, state = opt.update(grads, state, params)

    # Step 2: Main=0.25, Adam=0.81
    updates2, state = opt.update(grads, state, params)

    # Check that Muon updates decayed much faster than Adam updates
    # Muon update magnitude at step 2 vs step 0
    muon_ratio = (
        jnp.mean(jnp.abs(updates2["w"])) / jnp.mean(jnp.abs(updates0["w"]))
    )
    adam_ratio = (
        jnp.mean(jnp.abs(updates2["b"])) / jnp.mean(jnp.abs(updates0["b"]))
    )

    # Muon should have dropped more significantly
    self.assertLess(muon_ratio, adam_ratio)

  def test_muon_adamw_lr_with_weight_decay_mask(self):
    """Test interaction with weight_decay_mask."""
    params = {
        "w": jnp.ones((10, 10)),    # Muon
        "b": jnp.ones(10),          # Adam
    }
    # Mask out 'b' from weight decay
    def weight_decay_mask(params):
        return {"w": True, "b": False}
    opt = _muon.muon(
        learning_rate=0.01,
        learning_rate_adam=0.001,
        weight_decay=0.1,
        weight_decay_mask=weight_decay_mask
    )

    # Use zero gradients to isolate weight decay
    grads = jax.tree.map(jnp.zeros_like, params)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    # 'w' should have weight decay applied (non-zero update)
    # Note: Muon applies weight decay * learning_rate
    self.assertFalse(jnp.allclose(updates["w"], 0.0))
    # 'b' should NOT have weight decay (zero update because grad is zero)
    test_utils.assert_trees_all_close(
        updates["b"], jnp.zeros_like(updates["b"])
    )

  def test_muon_adamw_lr_validation(self):
    """Test that negative learning_rate_adam raises ValueError."""
    with self.assertRaisesRegex(
        ValueError, "learning_rate_adam must be non-negative"
    ):
        _muon.muon(learning_rate=0.01, learning_rate_adam=-0.001)


if __name__ == "__main__":
  absltest.main()
