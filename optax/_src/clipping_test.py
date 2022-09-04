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
"""Tests for `clipping.py`."""

from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp

from optax._src import clipping
from optax._src import linear_algebra

STEPS = 50
LR = 1e-2


class ClippingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  def test_clip(self):
    updates = self.per_step_updates
    # For a sufficiently high delta the update should not be changed.
    clipper = clipping.clip(1e6)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_tree_all_close(clipped_updates, clipped_updates)
    # Clipping at delta=1 should make all updates exactly 1.
    clipper = clipping.clip(1.)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_tree_all_close(
        clipped_updates, jax.tree_util.tree_map(jnp.ones_like, updates))

  def test_clip_by_block_rms(self):
    rmf_fn = lambda t: jnp.sqrt(jnp.mean(t**2))
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = clipping.clip_by_block_rms(1. / i)
      # Check that the clipper actually works and block rms is <= threshold
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(rmf_fn(updates[0]), 1. / i)
      self.assertAlmostEqual(rmf_fn(updates[1]), 1. / i)
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      chex.assert_tree_all_close(updates, updates_step)

  def test_clip_by_global_norm(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = clipping.clip_by_global_norm(1. / i)
      # Check that the clipper actually works and global norm is <= max_norm
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(
          linear_algebra.global_norm(updates), 1. / i, places=6)
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      chex.assert_tree_all_close(updates, updates_step)

  def test_adaptive_grad_clip(self):
    updates = self.per_step_updates
    params = self.init_params
    for i in range(1, STEPS + 1):
      clip_r = 1. / i
      clipper = clipping.adaptive_grad_clip(clip_r)

      # Check that the clipper actually works and upd_norm is < c * param_norm.
      updates, _ = clipper.update(updates, None, params)
      u_norm, p_norm = jax.tree_util.tree_map(
          clipping.unitwise_norm, (updates, params))
      cmp = jax.tree_util.tree_map(
          lambda u, p, c=clip_r: u - c * p < 1e-6, u_norm, p_norm)
      for leaf in jax.tree_util.tree_leaves(cmp):
        self.assertTrue(leaf.all())

      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None, params)
      chex.assert_tree_all_close(updates, updates_step)


if __name__ == '__main__':
  absltest.main()
