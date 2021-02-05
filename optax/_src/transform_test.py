# Lint as: python3
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
"""Tests for `transform.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update

STEPS = 50
LR = 1e-2


class TransformTest(parameterized.TestCase):

  def setUp(self):
    super(TransformTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants
  @parameterized.named_parameters([
      ('adam', transform.scale_by_adam),
      ('rmsprop', transform.scale_by_rms),
      ('fromage', transform.scale_by_fromage),
      ('stddev', transform.scale_by_stddev),
      ('trust_ratio', transform.scale_by_trust_ratio),
  ])
  def test_scalers(self, scaler_constr):
    params = self.init_params

    scaler = scaler_constr()
    init_fn = self.variant(scaler.init)
    transform_fn = self.variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    updates, state = transform_fn(self.per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))
    jax.tree_multimap(lambda *args: chex.assert_equal_shape(args), params,
                      updates)

  @chex.all_variants()
  def test_scale_by_fromage(self):
    schedule = lambda c: 1.0/(c + 1.0)
    fromage = transform.scale_by_fromage(step_size_factor_fn=schedule)
    params = self.init_params
    state = fromage.init(params)
    previous_norm = jnp.inf
    transform_fn = self.variant(fromage.update)

    for _ in range(STEPS):
      # Apply a step of fromage
      updates, state = transform_fn(self.per_step_updates, state, params)
      # Updates should get smaller due to the the learning schedule.
      norm = transform.global_norm(updates)
      self.assertLess(norm, previous_norm)
      previous_norm = norm

  @chex.all_variants()
  def test_apply_every(self):
    # The frequency of the application of sgd
    k = 4
    zero_update = (jnp.array([0., 0.]), jnp.array([0., 0.]))

    # optax sgd
    optax_sgd_params = self.init_params
    sgd = alias.sgd(LR, 0.0)
    state_sgd = sgd.init(optax_sgd_params)

    # optax sgd plus apply every
    optax_sgd_apply_every_params = self.init_params
    sgd_apply_every = combine.chain(
        transform.apply_every(k=k),
        transform.trace(decay=0, nesterov=False),
        transform.scale(-LR))
    state_sgd_apply_every = sgd_apply_every.init(optax_sgd_apply_every_params)
    transform_fn = self.variant(sgd_apply_every.update)

    for i in range(STEPS):
      # Apply a step of sgd
      updates_sgd, state_sgd = sgd.update(self.per_step_updates, state_sgd)
      optax_sgd_params = update.apply_updates(optax_sgd_params, updates_sgd)

      # Apply a step of sgd_apply_every
      updates_sgd_apply_every, state_sgd_apply_every = transform_fn(
          self.per_step_updates, state_sgd_apply_every)
      optax_sgd_apply_every_params = update.apply_updates(
          optax_sgd_apply_every_params, updates_sgd_apply_every)

      # Every k steps, check equivalence.
      if i % k == k-1:
        chex.assert_tree_all_close(
            optax_sgd_apply_every_params, optax_sgd_params,
            atol=1e-6, rtol=1e-5)
      # Otherwise, check update is zero.
      else:
        chex.assert_tree_all_close(
            updates_sgd_apply_every, zero_update, atol=0.0, rtol=0.0)

  def test_scale(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      factor = 0.1 ** i
      rescaler = transform.scale(factor)
      # Apply rescaling.
      scaled_updates, _ = rescaler.update(updates, None)
      # Manually scale updates.
      def rescale(t):
        return t * factor  # pylint:disable=cell-var-from-loop
      manual_updates = jax.tree_map(rescale, updates)
      # Check the rescaled updates match.
      chex.assert_tree_all_close(scaled_updates, manual_updates)

  def test_clip(self):
    updates = self.per_step_updates
    # For a sufficiently high delta the update should not be changed.
    clipper = transform.clip(1e6)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_tree_all_close(clipped_updates, clipped_updates)
    # Clipping at delta=1 should make all updates exactly 1.
    clipper = transform.clip(1.)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_tree_all_close(
        clipped_updates, jax.tree_map(jnp.ones_like, updates))

  def test_clip_by_global_norm(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = transform.clip_by_global_norm(1. / i)
      updates, _ = clipper.update(updates, None)
      # Check that the clipper actually works and global norm is <= max_norm
      self.assertAlmostEqual(transform.global_norm(updates), 1. / i, places=6)
      updates_step, _ = clipper.update(self.per_step_updates, None)
      # Check that continuously clipping won't cause numerical issues.
      chex.assert_tree_all_close(updates, updates_step, atol=1e-7, rtol=1e-7)

  @parameterized.named_parameters([
      ('1d', [1.0, 2.0], [1.0, 2.0]),
      ('2d', [[1.0, 2.0], [3.0, 4.0]], [[-0.5, 0.5], [-0.5, 0.5]]),
      ('3d', [[[1., 2.], [3., 4.]],
              [[5., 6.], [7., 8.]]], [[[-1.5, -0.5], [0.5, 1.5]],
                                      [[-1.5, -0.5], [0.5, 1.5]]]),
  ])
  def test_centralize(self, inputs, outputs):
    inputs = jnp.asarray(inputs)
    outputs = jnp.asarray(outputs)
    centralizer = transform.centralize()
    centralized_inputs, _ = centralizer.update(inputs, None)
    chex.assert_tree_all_close(centralized_inputs, outputs)

if __name__ == '__main__':
  absltest.main()
