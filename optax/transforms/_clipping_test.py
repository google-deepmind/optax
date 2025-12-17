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
"""Tests for methods in `optax.transforms._clipping.py`."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import test_utils
from optax.transforms import _clipping
import optax.tree


STEPS = 50
LR = 1e-2


class ClippingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (
        jnp.array([500.0, 5.0]),
        jnp.array([300.0, 3.0]),
    )

  def test_clip(self):
    updates = self.per_step_updates
    # For a sufficiently high delta the update should not be changed.
    clipper = _clipping.clip(1e6)
    clipped_updates, _ = clipper.update(updates, None)
    test_utils.assert_trees_all_close(clipped_updates, updates)

    # Clipping at delta=1 should make all updates exactly 1.
    clipper = _clipping.clip(1.0)
    clipped_updates, _ = clipper.update(updates, None)
    test_utils.assert_trees_all_close(
        clipped_updates, jax.tree.map(jnp.ones_like, updates)
    )

  def test_clip_by_block_rms(self):
    rmf_fn = lambda t: jnp.sqrt(jnp.mean(t**2))
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = _clipping.clip_by_block_rms(1.0 / i)
      # Check that the clipper actually works and block rms is <= threshold
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(rmf_fn(updates[0]), 1.0 / i)
      self.assertAlmostEqual(rmf_fn(updates[1]), 1.0 / i)
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      test_utils.assert_trees_all_close(updates, updates_step)

  def test_clip_by_global_norm(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = _clipping.clip_by_global_norm(1.0 / i)
      # Check that the clipper actually works and global norm is <= max_norm
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(optax.tree.norm(updates), 1.0 / i, places=6)
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      test_utils.assert_trees_all_close(updates, updates_step)

  def test_adaptive_grad_clip_with_axis(self):
    """Test adaptive_grad_clip with custom axis parameter."""

    # Test case 1: 5D Conv3D kernel (H, W, D, in_channels, out_channels)
    conv3d_grad = jnp.ones((2, 2, 2, 3, 4)) * 2.0  # Shape: (2,2,2,3,4)
    conv3d_param = jnp.ones((2, 2, 2, 3, 4))

    # Test case 2: Regular 2D weight matrix for comparison
    linear_grad = jnp.ones((5, 6)) * 3.0
    linear_param = jnp.ones((5, 6))

    # For Conv3D, we want to compute norms over spatial dimensions (0,1,2)
    # This gives us per-channel-pair norms of shape (1,1,1,3,4)
    agc_conv3d = _clipping.adaptive_grad_clip(clipping=0.5, axis=(0, 1, 2))

    # For linear layer, use default behavior (should work with 2D)
    agc_linear = _clipping.adaptive_grad_clip(clipping=0.5)

    # Test Conv3D clipping
    clipped_conv3d, _ = agc_conv3d.update([conv3d_grad], None, [conv3d_param])
    clipped_conv3d = clipped_conv3d[0]

    # Test linear layer clipping
    clipped_linear, _ = agc_linear.update([linear_grad], None, [linear_param])
    clipped_linear = clipped_linear[0]

    # Verify shapes are preserved
    self.assertEqual(clipped_conv3d.shape, conv3d_grad.shape)
    self.assertEqual(clipped_linear.shape, linear_grad.shape)

    # Verify AGC constraint: ||clipped_grad|| <= clipping * ||param|| (unitwise)
    conv3d_grad_norm = _clipping.unitwise_norm(clipped_conv3d, axis=(0, 1, 2))
    conv3d_param_norm = _clipping.unitwise_norm(conv3d_param, axis=(0, 1, 2))

    linear_grad_norm = _clipping.unitwise_norm(clipped_linear)
    linear_param_norm = _clipping.unitwise_norm(linear_param)

    # Check AGC constraint (with small tolerance for numerical precision)
    max_allowed_conv3d = 0.5 * conv3d_param_norm + 1e-6
    max_allowed_linear = 0.5 * linear_param_norm + 1e-6

    self.assertTrue(jnp.all(conv3d_grad_norm <= max_allowed_conv3d))
    self.assertTrue(jnp.all(linear_grad_norm <= max_allowed_linear))

    # Test that without clipping needed, gradients are unchanged
    small_conv3d_grad = jnp.ones((2, 2, 2, 3, 4)) * 0.1  # Small gradient
    small_conv3d_param = jnp.ones((2, 2, 2, 3, 4))

    agc_no_clip = _clipping.adaptive_grad_clip(clipping=2.0, axis=(0, 1, 2))
    unclipped, _ = agc_no_clip.update(
        [small_conv3d_grad], None, [small_conv3d_param]
    )

    # Should be nearly unchanged since clipping=2.0 is large
    test_utils.assert_trees_all_close(
        unclipped[0], small_conv3d_grad, rtol=1e-6)

  def test_unitwise_norm_with_axis(self):
    """Test unitwise_norm function with custom axis parameter."""

    # Test 5D tensor
    x = jnp.arange(2 * 2 * 2 * 3 * 4).reshape(2, 2, 2, 3, 4).astype(jnp.float32)

    # Test different axis configurations
    norm_spatial = _clipping.unitwise_norm(
        x, axis=(0, 1, 2)
    )  # Over spatial dims
    norm_channels = _clipping.unitwise_norm(x, axis=(3, 4))  # Over channel dims
    norm_last = _clipping.unitwise_norm(x, axis=-1)  # Over last dim only

    # Verify shapes
    self.assertEqual(norm_spatial.shape, x.shape)  # Should broadcast back
    self.assertEqual(norm_channels.shape, x.shape)
    self.assertEqual(norm_last.shape, x.shape)

    # Test that default behavior still works for supported shapes
    x_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    norm_default = _clipping.unitwise_norm(x_2d)
    norm_explicit = _clipping.unitwise_norm(x_2d, axis=None)

    test_utils.assert_trees_all_close(norm_default, norm_explicit)

    # Test error case: unsupported shape without axis
    x_6d = jnp.ones((2, 2, 2, 2, 2, 2))
    with self.assertRaises(ValueError):
      _clipping.unitwise_norm(x_6d)  # Should fail without axis

    # But should work with axis specified
    norm_6d = _clipping.unitwise_norm(x_6d, axis=(0, 1, 2))
    self.assertEqual(norm_6d.shape, x_6d.shape)

  def test_adaptive_grad_clip(self):
    updates = self.per_step_updates
    params = self.init_params
    for i in range(1, STEPS + 1):
      clip_r = 1.0 / i
      clipper = _clipping.adaptive_grad_clip(clipping=clip_r)

      # Check that the clipper works and upd_norm is < clip_r * param_norm.
      # (Line split to comply with line length limit)
      updates, _ = clipper.update(updates, None, params)
      u_norm, p_norm = jax.tree.map(_clipping.unitwise_norm, (updates, params))
      cmp = jax.tree.map(
          lambda u, p, c=clip_r: (u - c * p < 1e-6).all(), u_norm, p_norm
      )
      for leaf in jax.tree.leaves(cmp):
        self.assertTrue(leaf)

      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None, params)
      test_utils.assert_trees_all_close(updates, updates_step)

    # Simple 2D tensor case for default behavior
    grads_small = [jnp.array([[1.0, 2.0], [3.0, 4.0]])]
    params_small = [jnp.ones_like(grads_small[0])]
    agc_default = _clipping.adaptive_grad_clip(clipping=1.0)
    clipped_small, _ = agc_default.update(grads_small, None, params_small)
    clipped_norm_small = _clipping.unitwise_norm(clipped_small[0])
    param_norm_small = _clipping.unitwise_norm(params_small[0])
    self.assertTrue(
        jnp.all(clipped_norm_small <= 1.0 * param_norm_small + 1e-6)
    )

  def test_per_example_global_norm_clip(self):
    grads = [  # 3 users, 2 components
        jnp.array(
            [
                [0, -0.5],  # norm = sqrt(0^2 + 0.5^2 + 0^2)
                [3, 4],  # norm = sqrt(3^2 + 4^2 + 5^2)
                [5, 6],  # norm = sqrt(5^2 + 6^2 + 3^2)
                [0, 0],  # norm = 0
            ]
        ),
        jnp.array([[0], [5], [-3], [0]]),
    ]
    answer = [
        jnp.array([0, -0.5])
        + jnp.array([3, 4]) / jnp.sqrt(50)
        + jnp.array([5, 6]) / jnp.sqrt(70),
        jnp.array([0])
        + jnp.array([5]) / jnp.sqrt(50)
        + jnp.array([-3]) / jnp.sqrt(70),
    ]
    sum_clipped_grads, num_clipped = _clipping.per_example_global_norm_clip(
        grads, l2_norm_clip=1.0
    )

    for actual, expected in zip(sum_clipped_grads, answer):
      np.testing.assert_allclose(actual, expected, atol=1e-6)
    self.assertEqual(num_clipped, 2)

  def test_per_example_layer_norm_clip(self):
    # Test data for a model with two layers and a batch size of 4. The
    # 0th layer has one parameter (shape (1)), and the 1st layer has shape
    # (3, 3, 2).
    grads_flat = [
        jnp.array([[0.5], [1.5], [-2.0], [3.0]]),
        jnp.ones([4, 3, 3, 2], dtype=jnp.float32),
    ]

    with self.subTest(name="Uniform Variant"):
      sum_clipped_grads, num_clipped = _clipping.per_example_layer_norm_clip(
          grads_flat, global_l2_norm_clip=jnp.sqrt(2), uniform=True
      )

      # For the uniform variant, with global_l2_norm_clip=sqrt(2), the per-layer
      # clip norm is 1.0. Thus the per-example per-layer clipped grads are
      # [[0.5], [1.0], [-1.0], [1.0]] and [1 / sqrt(18) ... ]. The sum of
      # these over the 4 input gradients are [1.5] and [4 / sqrt(18) ...].
      self.assertAlmostEqual(sum_clipped_grads[0], 1.5)
      for element in sum_clipped_grads[1].flatten():
        self.assertAlmostEqual(element, 4 / jnp.sqrt(18), places=4)

      # The three values in grads_flat[0] with magnitude > 1.0 are clipped, as
      # are all four values in grads_flat[1].
      self.assertEqual(num_clipped[0], 3)
      self.assertEqual(num_clipped[1], 4)

    with self.subTest(name="Scaled Variant"):
      sum_clipped_grads, num_clipped = _clipping.per_example_layer_norm_clip(
          grads_flat, global_l2_norm_clip=jnp.sqrt(19), uniform=False
      )

      # For the scaled variant, with global_l2_norm_clip=sqrt(19), the per-layer
      # clip norm for the 0th layer is 1.0, and the per-layer clip norm for
      # the 1st layer is sqrt(18). Thus the per-example per-layer clipped grads
      # are [[0.5], [1.0], [-1.0], [1.0]] and [[1.0)] ... ]. The sum of
      # these over the 4 input gradients are [1.5] and [4.0 ...].
      self.assertAlmostEqual(sum_clipped_grads[0], 1.5)
      for element in sum_clipped_grads[1].flatten():
        self.assertAlmostEqual(element, 4.0)

      # The three values in grads_flat[0] with magnitude > 1.0 are clipped. The
      # grad norms for grads_flat[1] are all equal to the per-layer clip norm,
      # so none of these grads are clipped.
      self.assertEqual(num_clipped[0], 3)
      self.assertEqual(num_clipped[1], 0)


if __name__ == "__main__":
  absltest.main()
