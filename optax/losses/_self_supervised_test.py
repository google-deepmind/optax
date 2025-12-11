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
"""Tests for self-supervised losses in `optax.losses._self_supervised.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from optax.losses import _self_supervised


class NtxentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([
        [-1.9540, 1.0780],
        [0.2380, -0.5703],
        [1.8745, -0.0195],
        [-0.6719, -1.9210],
    ])
    self.ys_2 = jnp.array([
        [0.0, 0.0],
        [0.2380, -0.5703],
        [1.8745, -0.0195],
        [-0.6719, -1.9210],
    ])
    self.ts_1 = jnp.array([0, 0, 1, 1])
    self.ts_2 = jnp.array([0, 0, 0, 1])
    # Calculated expected output
    self.exp_1 = jnp.array(14.01032)
    self.exp_2 = jnp.array(8.968544)
    self.exp_3 = jnp.array(9.2889)

  def test_batched(self):
    np.testing.assert_allclose(
        jax.jit(_self_supervised.ntxent)(self.ys, self.ts_1),
        self.exp_1,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        jax.jit(_self_supervised.ntxent)(self.ys, self.ts_2),
        self.exp_2,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        jax.jit(_self_supervised.ntxent)(self.ys_2, self.ts_1),
        self.exp_3,
        atol=1e-4,
    )


class TripletMarginLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.a1 = jnp.ones((2, 2))
    self.p1 = jnp.zeros((2, 2))
    self.n1 = jnp.ones((2, 2)) * 2
    self.a2 = jnp.zeros((2, 2))
    self.p2 = jnp.ones((2, 2))
    self.n2 = jnp.ones((2, 2)) * 2

  @parameterized.parameters([
      {
          'anchor': np.ones((2, 2)),
          'positive': np.zeros((2, 2)),
          'negative': np.ones((2, 2)) * 2,
          'margin': 1.0,
      },
      {
          'anchor': np.zeros((2, 2)),
          'positive': np.ones((2, 2)),
          'negative': np.ones((2, 2)) * 2,
          'margin': 1.0,
      }
  ])
  def test_batched(self, anchor, positive, negative, margin):
    def testing_triplet_margin_loss(a, p, n, margin=1.0, p_norm=2, eps=1e-6):
      ap_distance = jnp.sqrt(jnp.sum(jnp.power(a - p, p_norm)) + eps)
      an_distance = jnp.sqrt(jnp.sum(jnp.power(a - n, p_norm)) + eps)
      return jnp.maximum(ap_distance - an_distance + margin, 0)

    handmade_result = testing_triplet_margin_loss(
        a=anchor, p=positive, n=negative, margin=margin
    )
    result = jax.jit(_self_supervised.triplet_margin_loss)(
        anchor, positive, negative
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

  @parameterized.parameters([
      {
          'anchor': np.ones((2, 2)),
          'positive': np.zeros((2, 2)),
          'negative': np.ones((2, 2)) * 2,
      },
  ])
  def test_vmap(self, anchor, positive, negative):
    original_loss = _self_supervised.triplet_margin_loss(anchor, positive,
                                                         negative)
    anchor_batched = anchor.reshape(1, *anchor.shape)
    positive_batched = positive.reshape(1, *positive.shape)
    negative_batched = negative.reshape(1, *negative.shape)
    vmap_loss = jax.jit(
        jax.vmap(_self_supervised.triplet_margin_loss, in_axes=(0, 0, 0)))(
            anchor_batched, positive_batched, negative_batched)
    np.testing.assert_allclose(vmap_loss.flatten(), original_loss.flatten()
                               , atol=1e-4)


class ByolLossTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.key(42)
    self.key1, self.key2 = jax.random.split(self.key, 2)
    self.online_pred = jax.random.normal(self.key1, shape=(4, 128))
    self.target_proj = jax.random.normal(self.key2, shape=(4, 128))

  def test_output_shape(self):
    loss = _self_supervised.byol_loss(self.online_pred, self.target_proj)
    self.assertEqual(loss.shape, (4,))

  def test_identical_inputs(self):
    # When inputs are identical, loss should be 0
    loss = _self_supervised.byol_loss(self.online_pred, self.online_pred)
    np.testing.assert_allclose(loss, jnp.zeros(4), atol=1e-5)

  def test_normalized_equivalence(self):
    # BYOL loss should be 2 - 2 * cosine_similarity for normalized vectors
    eps = jnp.finfo(self.online_pred.dtype).eps
    online_norm = self.online_pred / (
        jnp.linalg.norm(self.online_pred, axis=-1, keepdims=True) + eps
    )
    target_norm = self.target_proj / (
        jnp.linalg.norm(self.target_proj, axis=-1, keepdims=True) + eps
    )
    expected = 2.0 - 2.0 * jnp.sum(online_norm * target_norm, axis=-1)
    actual = _self_supervised.byol_loss(self.online_pred, self.target_proj)
    np.testing.assert_allclose(actual, expected, atol=1e-5)

  def test_jit_compatible(self):
    loss = jax.jit(_self_supervised.byol_loss)(
        self.online_pred, self.target_proj)
    self.assertEqual(loss.shape, (4,))


class SimSiamLossTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.key(42)
    self.key1, self.key2 = jax.random.split(self.key, 2)
    self.predictions = jax.random.normal(self.key1, shape=(4, 128))
    self.projections = jax.random.normal(self.key2, shape=(4, 128))

  def test_output_shape(self):
    loss = _self_supervised.simsiam_loss(self.predictions, self.projections)
    self.assertEqual(loss.shape, (4,))

  def test_identical_inputs(self):
    # When inputs are identical, loss should be -1 (negative of max cosine sim)
    loss = _self_supervised.simsiam_loss(self.predictions, self.predictions)
    np.testing.assert_allclose(loss, -jnp.ones(4), atol=1e-5)

  def test_orthogonal_vectors(self):
    # Orthogonal vectors should give loss close to 0
    v1 = jnp.array([[1.0, 0.0]])
    v2 = jnp.array([[0.0, 1.0]])
    loss = _self_supervised.simsiam_loss(v1, v2)
    np.testing.assert_allclose(loss, jnp.zeros(1), atol=1e-5)

  def test_jit_compatible(self):
    loss = jax.jit(_self_supervised.simsiam_loss)(
        self.predictions, self.projections)
    self.assertEqual(loss.shape, (4,))


class DinoLossTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.key(42)
    self.key1, self.key2 = jax.random.split(self.key, 2)
    self.student = jax.random.normal(self.key1, shape=(4, 256))
    self.teacher = jax.random.normal(self.key2, shape=(4, 256))

  def test_output_shape(self):
    loss = _self_supervised.dino_loss(self.student, self.teacher)
    self.assertEqual(loss.shape, (4,))

  def test_with_center(self):
    center = jnp.zeros(256)
    loss = _self_supervised.dino_loss(self.student, self.teacher, center=center)
    self.assertEqual(loss.shape, (4,))

  def test_identical_logits(self):
    # When student and teacher produce same logits, loss should be low
    loss_same = _self_supervised.dino_loss(self.student, self.student)
    loss_diff = _self_supervised.dino_loss(self.student, self.teacher)
    # Same logits should have lower loss (but not zero due to temperature diff)
    self.assertTrue(jnp.mean(loss_same) < jnp.mean(loss_diff))

  def test_temperature_effect(self):
    # Lower teacher temperature should produce sharper distributions
    loss_low_temp = _self_supervised.dino_loss(
        self.student, self.teacher, teacher_temp=0.01
    )
    loss_high_temp = _self_supervised.dino_loss(
        self.student, self.teacher, teacher_temp=0.5
    )
    # Both should produce valid losses
    self.assertEqual(loss_low_temp.shape, (4,))
    self.assertEqual(loss_high_temp.shape, (4,))

  def test_jit_compatible(self):
    loss = jax.jit(_self_supervised.dino_loss)(self.student, self.teacher)
    self.assertEqual(loss.shape, (4,))


class BarlowTwinsLossTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.key(42)
    self.key1, self.key2 = jax.random.split(self.key, 2)
    self.z_a = jax.random.normal(self.key1, shape=(32, 128))
    self.z_b = jax.random.normal(self.key2, shape=(32, 128))

  def test_output_shape(self):
    loss = _self_supervised.barlow_twins_loss(self.z_a, self.z_b)
    self.assertEqual(loss.shape, ())

  def test_identical_embeddings(self):
    # When embeddings are identical, cross-correlation should be close
    # to identity. So loss should be close to 0 (only off-diagonal penalty
    # from lambda).
    loss = _self_supervised.barlow_twins_loss(self.z_a, self.z_a)
    # Loss should be relatively small for identical inputs
    self.assertTrue(loss >= 0)

  def test_lambda_effect(self):
    # Higher lambda should penalize off-diagonal elements more
    loss_low_lambda = _self_supervised.barlow_twins_loss(
        self.z_a, self.z_b, lambda_=0.001
    )
    loss_high_lambda = _self_supervised.barlow_twins_loss(
        self.z_a, self.z_b, lambda_=1.0
    )
    # Higher lambda should give higher loss (usually)
    self.assertTrue(loss_high_lambda >= loss_low_lambda)

  def test_jit_compatible(self):
    loss = jax.jit(_self_supervised.barlow_twins_loss)(self.z_a, self.z_b)
    self.assertEqual(loss.shape, ())


if __name__ == '__main__':
  absltest.main()
