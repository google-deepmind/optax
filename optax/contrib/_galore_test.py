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
"""Tests for GaLore optimizer (optax.contrib._galore)."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import optax

from optax.contrib import _galore


def _tree_sum_squares(tree):
  leaves = jax.tree.leaves(tree)
  return sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves)


class GaLoreTest(parameterized.TestCase):
  """Comprehensive tests for GaLore optimizer.

  These tests validate:
  - left vs right projection selection
  - low-rank memory reduction
  - projector orthonormality
  - projector update cadence
  - correctness under jit
  - robustness for mixed pytrees
  - dtype behavior (bf16 + mu_dtype)
  - convergence on simple objectives
  """

  # ---------------------------------------------------------------------------
  # Shape & state structure tests
  # ---------------------------------------------------------------------------

  @parameterized.parameters(
      # Right projection when m < n (moments are m×r, smaller than r×n)
      {"shape": (32, 64), "rank": 8, "expect_left": False},
      # Left projection when m > n (moments are r×n, smaller than m×r)
      {"shape": (64, 32), "rank": 8, "expect_left": True},
      # Square defaults to left
      {"shape": (64, 64), "rank": 8, "expect_left": True},
  )
  def test_left_vs_right_projection_state_shapes(self,shape, rank, expect_left):
    params = jnp.zeros(shape, dtype=jnp.float32)
    opt = _galore.galore(learning_rate=0.1, rank=rank, update_proj_gap=10)
    state = opt.init(params)

    galore_state = state[0]
    # Access mu/nu from the base optimizer state (default is scale_by_adam)
    base_state = galore_state.base_optimizer_state
    mu, nu, P = base_state.mu, base_state.nu, galore_state.projector

    m_dim, n_dim = shape
    r_eff = min(rank, m_dim, n_dim)

    if expect_left:
      self.assertEqual(P.shape, (m_dim, r_eff))
      self.assertEqual(mu.shape, (r_eff, n_dim))
      self.assertEqual(nu.shape, (r_eff, n_dim))
    else:
      self.assertEqual(P.shape, (n_dim, r_eff))
      self.assertEqual(mu.shape, (m_dim, r_eff))
      self.assertEqual(nu.shape, (m_dim, r_eff))

  def test_rank_is_clipped_to_min_dimension(self):
    params = jnp.zeros((10, 3), dtype=jnp.float32)
    opt = _galore.galore(learning_rate=0.1, rank=999, update_proj_gap=10)
    state = opt.init(params)
    galore_state = state[0]
    base_state = galore_state.base_optimizer_state

    # min(m,n)=3, m>n => left projection (moments r×n = 3×3)
    self.assertEqual(galore_state.projector.shape, (10, 3))  # (m, r)
    self.assertEqual(base_state.mu.shape, (3, 3))            # (r, n)
    self.assertEqual(base_state.nu.shape, (3, 3))            # (r, n)

  # ---------------------------------------------------------------------------
  # Memory reduction tests
  # ---------------------------------------------------------------------------

  def test_memory_reduction_vs_full_moments(self):
    dim = 256
    rank = 8
    params = jnp.zeros((dim, dim), dtype=jnp.float32)
    opt = _galore.galore(learning_rate=0.1, rank=rank)
    state = opt.init(params)
    galore_state = state[0]
    base_state = galore_state.base_optimizer_state

    self.assertEqual(base_state.mu.size, rank * dim)
    self.assertEqual(base_state.nu.size, rank * dim)
    self.assertLess(base_state.mu.size, dim * dim)

  # ---------------------------------------------------------------------------
  # Projector correctness tests
  # ---------------------------------------------------------------------------

  def test_projector_orthonormality_after_update(self):
    m, n, r = 16, 32, 4
    params = {"w": jax.random.normal(jax.random.key(0), (m, n))}
    opt = _galore.galore(learning_rate=0.1, rank=r, update_proj_gap=1)
    state = opt.init(params)

    grads = {"w": jax.random.normal(jax.random.key(1), (m, n))}
    _, state2 = opt.update(grads, state, params)

    P = state2[0].projector["w"]
    gram = P.T @ P
    self.assertTrue(
        jnp.allclose(gram, jnp.eye(r, dtype=gram.dtype), atol=1e-2, rtol=1e-2)
    )

  def test_projector_update_gap_behavior(self):
    params = jnp.ones((32, 64), dtype=jnp.float32)
    gap = 1000
    opt = _galore.galore(learning_rate=0.01, rank=8, update_proj_gap=gap)
    state = opt.init(params)

    grads0 = jnp.arange(params.size, dtype=jnp.float32).reshape(params.shape)
    _, s1 = opt.update(grads0, state, params)
    P0 = s1[0].projector

    grads1 = jnp.flip(grads0, axis=0)
    _, s2 = opt.update(grads1, s1, params)
    P1 = s2[0].projector

    self.assertTrue(jnp.allclose(P0, P1))

  def test_projector_updates_when_gap_is_one(self):
    params = jnp.ones((32, 64), dtype=jnp.float32)
    opt = _galore.galore(learning_rate=0.01, rank=8, update_proj_gap=1)
    state = opt.init(params)

    grads0 = jnp.arange(params.size, dtype=jnp.float32).reshape(params.shape)
    _, s1 = opt.update(grads0, state, params)
    P0 = s1[0].projector

    grads1 = jnp.flip(grads0, axis=0)
    _, s2 = opt.update(grads1, s1, params)
    P1 = s2[0].projector

    self.assertFalse(jnp.allclose(P0, P1))

  # ---------------------------------------------------------------------------
  # Optimization & convergence tests
  # ---------------------------------------------------------------------------

  def test_converges_on_simple_quadratic_2d(self):
    W = jnp.ones((64, 64))
    opt = _galore.galore(learning_rate=0.1, rank=8, update_proj_gap=1)
    state = opt.init(W)

    @jax.jit
    def step(p, s):
      g = p
      u, ns = opt.update(g, s, p)
      return optax.apply_updates(p, u), ns

    init_norm = jnp.linalg.norm(W)
    for _ in range(20):
      W, state = step(W, state)
    self.assertLess(jnp.linalg.norm(W), init_norm)

  def test_optimization_decreases_total_energy_for_pytree(self):
    params = {
        "W1": jnp.ones((32, 64)),
        "W2": jnp.ones((64, 16)),
        "b": jnp.ones((16,)),
    }
    opt = _galore.galore(learning_rate=0.05, rank=8, update_proj_gap=2)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
      grads = jax.tree.map(lambda x: x, p)
      u, ns = opt.update(grads, s, p)
      return optax.apply_updates(p, u), ns

    e0 = _tree_sum_squares(params)
    for _ in range(10):
      params, state = step(params, state)
    e1 = _tree_sum_squares(params)

    self.assertLess(e1, e0)

  # ---------------------------------------------------------------------------
  # Robustness & regression tests
  # ---------------------------------------------------------------------------

  def test_mixed_pytree_shapes_jittable_regression(self):
    params = {
        "W": jnp.ones((16, 32), dtype=jnp.bfloat16),
        "b": jnp.ones((32,), dtype=jnp.bfloat16),
        "conv": jnp.ones((3, 3, 8), dtype=jnp.bfloat16),
    }
    # Use base_optimizer with mu_dtype for controlling moment precision
    base_opt = optax.scale_by_adam(mu_dtype=jnp.float32)
    opt = _galore.galore(
        learning_rate=0.01,
        rank=4,
        update_proj_gap=5,
        base_optimizer=base_opt,
    )
    state = opt.init(params)

    @jax.jit
    def step(p, s):
      grads = jax.tree.map(lambda x: x, p)
      u, ns = opt.update(grads, s, p)
      return optax.apply_updates(p, u), ns

    new_params, _ = step(params, state)

    self.assertEqual(new_params["W"].shape, (16, 32))
    self.assertEqual(new_params["b"].shape, (32,))
    self.assertEqual(new_params["conv"].shape, (3, 3, 8))

  def test_mu_dtype_controls_moment_dtype(self):
    params = jnp.ones((16, 32), dtype=jnp.bfloat16)
    # Configure mu_dtype through the base optimizer
    base_opt = optax.scale_by_adam(mu_dtype=jnp.float32)
    opt = _galore.galore(
        learning_rate=0.01,
        rank=4,
        update_proj_gap=1,
        base_optimizer=base_opt,
    )
    state = opt.init(params)
    galore_state = state[0]
    base_state = galore_state.base_optimizer_state

    self.assertEqual(base_state.mu.dtype, jnp.float32)
    # Note: nu (second moment) is not controlled by mu_dtype in scale_by_adam
    # It uses the params dtype by default
    self.assertEqual(base_state.nu.dtype, jnp.bfloat16)

    grads = jnp.ones_like(params)
    updates, _ = opt.update(grads, state, params)
    self.assertEqual(updates.dtype, jnp.bfloat16)

  # ---------------------------------------------------------------------------
  # Non-2D array support (GaLoreDimensionNumbers) tests
  # ---------------------------------------------------------------------------

  def test_3d_attention_weights_with_dimension_numbers(self):
    """Test projecting 3D attention weights as 2D matrices."""
    # Attention weights: (embed_dim, num_heads, head_dim)
    embed_dim, num_heads, head_dim = 512, 8, 64
    params = {'attn': jnp.ones((embed_dim, num_heads, head_dim))}

    dim_nums = {
        'attn': _galore.GaLoreDimensionNumbers(
            reduction_axis=0,      # embed_dim
            output_axis=(1, 2),    # heads * head_dim
        )
    }

    opt = _galore.galore(
        learning_rate=0.01,
        rank=16,
        weight_dimension_numbers=dim_nums,
    )
    state = opt.init(params)
    grads = {'attn': jnp.ones((embed_dim, num_heads, head_dim)) * 0.1}

    updates, _ = opt.update(grads, state, params)

    # Check output shape matches input
    self.assertEqual(updates['attn'].shape, (embed_dim, num_heads, head_dim))

    # Verify projector has correct shape
    # Reshaped to (512, 512), use left projection since m >= n
    galore_state = state[0]
    expected_proj_shape = (embed_dim, 16)  # (m, rank)
    self.assertEqual(galore_state.projector['attn'].shape, expected_proj_shape)

  def test_3d_memory_reduction_with_dimension_numbers(self):
    """Verify memory reduction for 3D tensors reshaped to 2D."""
    embed_dim, num_heads, head_dim = 256, 8, 64
    rank = 16
    params = {'w': jnp.zeros((embed_dim, num_heads, head_dim))}

    dim_nums = {
        'w': _galore.GaLoreDimensionNumbers(
            reduction_axis=0,
            output_axis=(1, 2),
        )
    }

    opt = _galore.galore(
        learning_rate=0.1,
        rank=rank,
        weight_dimension_numbers=dim_nums,
    )
    state = opt.init(params)
    galore_state = state[0]
    base_state = galore_state.base_optimizer_state

    # Reshaped: (256, 512), m < n → right projection
    # Moments should be (m, rank) = (256, 16)
    self.assertEqual(base_state.mu['w'].shape, (embed_dim, rank))
    self.assertEqual(base_state.nu['w'].shape, (embed_dim, rank))

    # Verify memory savings
    full_size = embed_dim * num_heads * head_dim
    moment_size = base_state.mu['w'].size
    self.assertLess(moment_size, full_size)

  def test_dimension_numbers_convergence(self):
    """Test that optimization converges with dimension numbers."""
    embed_dim, num_heads, head_dim = 64, 4, 16
    params = {'attn': jnp.ones((embed_dim, num_heads, head_dim))}

    dim_nums = {
        'attn': _galore.GaLoreDimensionNumbers(
            reduction_axis=0,
            output_axis=(1, 2),
        )
    }

    opt = _galore.galore(
        learning_rate=0.1,
        rank=8,
        update_proj_gap=1,
        weight_dimension_numbers=dim_nums,
    )
    state = opt.init(params)

    @jax.jit
    def step(p, s):
      grads = jax.tree.map(lambda x: x, p)
      u, ns = opt.update(grads, s, p)
      return optax.apply_updates(p, u), ns

    e0 = _tree_sum_squares(params)
    for _ in range(10):
      params, state = step(params, state)
    e1 = _tree_sum_squares(params)

    self.assertLess(e1, e0)

  def test_mixed_2d_and_3d_params_with_dimension_numbers(self):
    """Test pytree with both 2D and 3D parameters."""
    params = {
        'attn': jnp.ones((128, 4, 32)),   # 3D attention
        'mlp': jnp.ones((128, 256)),       # 2D linear
        'bias': jnp.ones((256,)),          # 1D bias
    }

    dim_nums = {
        'attn': _galore.GaLoreDimensionNumbers(
            reduction_axis=0,
            output_axis=(1, 2),
        ),
        'mlp': None,   # Use default 2D projection
        'bias': None,  # Skip (1D)
    }

    opt = _galore.galore(
        learning_rate=0.01,
        rank=8,
        weight_dimension_numbers=dim_nums,
    )
    state = opt.init(params)
    grads = jax.tree.map(lambda x: x * 0.1, params)

    updates, _ = opt.update(grads, state, params)

    # Check all shapes preserved
    self.assertEqual(updates['attn'].shape, (128, 4, 32))
    self.assertEqual(updates['mlp'].shape, (128, 256))
    self.assertEqual(updates['bias'].shape, (256,))

  def test_dimension_numbers_right_projection(self):
    """Test dimension numbers with right projection (m < n)."""
    # Small input, large output → right projection
    params = {'w': jnp.ones((32, 8, 64))}  # reshaped to (32, 512)

    dim_nums = {
        'w': _galore.GaLoreDimensionNumbers(
            reduction_axis=0,      # 32
            output_axis=(1, 2),    # 8*64 = 512
        )
    }

    rank = 8
    opt = _galore.galore(
        learning_rate=0.01,
        rank=rank,
        weight_dimension_numbers=dim_nums,
    )
    state = opt.init(params)
    galore_state = state[0]
    base_state = galore_state.base_optimizer_state

    # m=32 < n=512 → right projection
    # Projector: (n, rank) = (512, 8)
    # Moments: (m, rank) = (32, 8)
    self.assertEqual(galore_state.projector['w'].shape, (512, rank))
    self.assertEqual(base_state.mu['w'].shape, (32, rank))

  def test_single_dimension_number_applied_to_all(self):
    """Test applying single GaLoreDimensionNumbers to all eligible params."""
    params = {
        'w1': jnp.ones((64, 4, 16)),
        'w2': jnp.ones((32, 8, 8)),
        'bias': jnp.ones((64,)),  # 1D, should be skipped
    }

    # Single dim_nums applied to all 2D+ params
    single_dim_nums = _galore.GaLoreDimensionNumbers(
        reduction_axis=0,
        output_axis=(1, 2),
    )

    opt = _galore.galore(
        learning_rate=0.01,
        rank=8,
        weight_dimension_numbers=single_dim_nums,
    )
    state = opt.init(params)
    grads = jax.tree.map(lambda x: x * 0.1, params)

    updates, _ = opt.update(grads, state, params)

    # All shapes preserved
    self.assertEqual(updates['w1'].shape, (64, 4, 16))
    self.assertEqual(updates['w2'].shape, (32, 8, 8))
    self.assertEqual(updates['bias'].shape, (64,))


if __name__ == "__main__":
  absltest.main()
