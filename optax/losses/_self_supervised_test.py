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
from optax.losses import _regression
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

class ByolLossTest(parameterized.TestCase):
  @parameterized.parameters(0, 1, 7, 42, 123)
  def test_symmetric_zero_for_identical_and_nonzero_otherwise(
    self, seed):
    key = jax.random.PRNGKey(seed)
    k_q1, k_q2, k_zrand = jax.random.split(key, 3)

    q1 = jax.random.normal(k_q1, (8, 16), dtype=jnp.float32)
    q2 = jax.random.normal(k_q2, (8, 16), dtype=jnp.float32)
    z2 = q1
    z1 = q2

    loss_identical = jax.jit(_self_supervised.byol_loss)(
        q1, z2, q2, z1, symmetric=True,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_identical))))
    np.testing.assert_allclose(loss_identical, 0.0, atol=1e-4)

    z2_random = jax.random.normal(k_zrand, z2.shape, dtype=jnp.float32)
    loss_random = jax.jit(_self_supervised.byol_loss)(
        q1, z2_random, q2, z1, symmetric=True,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_random))))
    self.assertGreater(float(loss_random), 1e-4)

  @parameterized.parameters(0, 1, 7, 42, 123)
  def test_single_direction_zero_for_identical_and_nonzero_otherwise(
    self, seed):
    key = jax.random.PRNGKey(seed)
    k_q, k_zrand = jax.random.split(key, 2)

    q = jax.random.normal(k_q, (8, 16), dtype=jnp.float32)
    z = q
    loss_identical = jax.jit(_self_supervised.byol_loss)(
        q, z,symmetric=False,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_identical))))
    np.testing.assert_allclose(loss_identical, 0.0, atol=1e-4)

    z_random = jax.random.normal(k_zrand, z.shape, dtype=jnp.float32)
    loss_random = jax.jit(_self_supervised.byol_loss)(
        q, z_random, symmetric=False,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_random))))
    self.assertGreater(float(loss_random), 1e-4)


class SimSiamLossTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 7, 42, 123)
  def test_symmetric_minimum_for_identical_and_higher_otherwise(
    self, seed):
    key = jax.random.PRNGKey(seed)
    k_p1, k_p2, k_zrand = jax.random.split(key, 3)

    p1 = jax.random.normal(k_p1, (8, 16), dtype=jnp.float32)
    p2 = jax.random.normal(k_p2, (8, 16), dtype=jnp.float32)
    z2 = p1
    z1 = p2

    loss_identical = jax.jit(_self_supervised.simsiam_loss)(
        p1, z2, p2, z1, symmetric=True,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_identical))))
    np.testing.assert_allclose(loss_identical, -1.0, atol=1e-4)

    z2_random = jax.random.normal(k_zrand, z2.shape, dtype=jnp.float32)
    loss_random = jax.jit(_self_supervised.simsiam_loss)(
        p1, z2_random, p2, z1, symmetric=True,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_random))))
    self.assertGreater(float(loss_random), float(loss_identical) + 1e-3)

  @parameterized.parameters(0, 1, 7, 42, 123)
  def test_single_direction_minimum_for_identical_and_higher_otherwise(
    self, seed):
    key = jax.random.PRNGKey(seed)
    k_p, k_zrand = jax.random.split(key, 2)
    p = jax.random.normal(k_p, (8, 16), dtype=jnp.float32)
    z = p
    loss_identical = jax.jit(_self_supervised.simsiam_loss)(
        p, z, symmetric=False,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_identical))))
    np.testing.assert_allclose(loss_identical, -1.0, atol=1e-4)

    z_random = jax.random.normal(k_zrand, z.shape, dtype=jnp.float32)
    loss_random = jax.jit(_self_supervised.simsiam_loss)(
        p, z_random, symmetric=False,
    )
    self.assertTrue(bool(np.isfinite(np.asarray(loss_random))))
    self.assertGreater(float(loss_random), float(loss_identical) + 1e-3)

class DinoLossTest(absltest.TestCase):

  def test_single_view_matches_handmade(self):
    student_logits = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.3, 0.1, 0.2]],
        dtype=jnp.float32,
    )
    teacher_logits = jnp.array(
        [[0.0, 0.0, 0.0],
         [0.2, 0.2, 0.0]],
        dtype=jnp.float32,
    )
    student_temperature = 0.1
    teacher_temperature = 0.04
    teacher_center = jnp.array(
        [0.1, -0.1, 0.0],
        dtype=jnp.float32,
    )

    def testing_dino_loss(
        student,
        teacher,
        student_temperature_val,
        teacher_temperature_val,
        teacher_center_val,
    ):
      teacher_scaled = (
          teacher - teacher_center_val
      ) / teacher_temperature_val
      student_scaled = student / student_temperature_val
      teacher_prob = jax.nn.softmax(teacher_scaled, axis=-1)
      log_student_prob = jax.nn.log_softmax(student_scaled, axis=-1)
      loss_per_example = -jnp.sum(
          teacher_prob * log_student_prob,
          axis=-1,
      )
      return jnp.mean(loss_per_example)

    handmade_result = testing_dino_loss(
        student_logits,
        teacher_logits,
        student_temperature,
        teacher_temperature,
        teacher_center,
    )
    result = jax.jit(_self_supervised.dino_loss)(
        student_logits,
        teacher_logits,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        teacher_center=teacher_center,
        two_view=False,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-6)

  def test_two_view_matches_handmade(self):
    s1 = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.3, 0.1, 0.2]],
        dtype=jnp.float32,
    )
    s2 = jnp.array(
        [[0.0, 0.1, 0.4],
         [0.2, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    t1 = jnp.array(
        [[0.0, 0.0, 0.0],
         [0.2, 0.2, 0.0]],
        dtype=jnp.float32,
    )
    t2 = jnp.array(
        [[0.1, -0.1, 0.0],
         [0.0, 0.1, -0.1]],
        dtype=jnp.float32,
    )
    student_temperature = 0.1
    teacher_temperature = 0.04
    teacher_center = jnp.array(
        [0.1, -0.1, 0.0],
        dtype=jnp.float32,
    )

    def single_view_dino(student, teacher):
      teacher_scaled = (teacher - teacher_center) / teacher_temperature
      student_scaled = student / student_temperature
      teacher_prob = jax.nn.softmax(teacher_scaled, axis=-1)
      log_student_prob = jax.nn.log_softmax(student_scaled, axis=-1)
      loss_per_example = -jnp.sum(teacher_prob * log_student_prob, axis=-1)
      return jnp.mean(loss_per_example)

    handmade_l12 = single_view_dino(s1, t2)
    handmade_l21 = single_view_dino(s2, t1)
    handmade_result = 0.5 * (handmade_l12 + handmade_l21)

    result = jax.jit(_self_supervised.dino_loss)(
        s1,
        t1,
        student_logits_2=s2,
        teacher_logits_2=t2,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        teacher_center=teacher_center,
        two_view=True,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-6)

  def test_temperatures_must_be_positive(self):
    student_logits = jnp.zeros((2, 3), dtype=jnp.float32)
    teacher_logits = jnp.zeros((2, 3), dtype=jnp.float32)

    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          student_temperature=0.0,
      )

    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          teacher_temperature=-0.1,
      )


class BarlowTwinsLossTest(absltest.TestCase):

  def test_batched_matches_handmade(self):
    z1 = jnp.array(
        [[0.1, -0.2, 0.3],
         [0.0, 0.5, -0.4],
         [1.0, -1.0, 0.2],
         [-0.3, 0.7, 0.1]],
        dtype=jnp.float32,
    )
    z2 = jnp.array(
        [[0.0, -0.1, 0.4],
         [0.2, 0.4, -0.3],
         [0.9, -0.8, 0.3],
         [-0.2, 0.6, 0.0]],
        dtype=jnp.float32,
    )

    def testing_barlow_twins_loss(
        z1_val,
        z2_val,
        off_diagonal_scale=5e-3,
        eps=1e-12,
    ):
      batch_size, feature_dim = z1_val.shape

      z1_mean = jnp.mean(z1_val, axis=0)
      z2_mean = jnp.mean(z2_val, axis=0)
      z1_centered = z1_val - z1_mean
      z2_centered = z2_val - z2_mean

      z1_var = jnp.mean(z1_centered ** 2, axis=0)
      z2_var = jnp.mean(z2_centered ** 2, axis=0)
      z1_std = jnp.sqrt(z1_var + eps)
      z2_std = jnp.sqrt(z2_var + eps)

      z1_norm = z1_centered / z1_std
      z2_norm = z2_centered / z2_std

      cross_correlation = z1_norm.T @ z2_norm
      cross_correlation /= batch_size

      on_diag = jnp.diag(cross_correlation)
      on_diag_loss = jnp.sum((1.0 - on_diag) ** 2)

      off_diag_mask = 1.0 - jnp.eye(
          feature_dim,
          dtype=cross_correlation.dtype,
      )
      off_diag = cross_correlation * off_diag_mask
      off_diag_loss = jnp.sum(off_diag ** 2)

      return on_diag_loss + off_diagonal_scale * off_diag_loss

    handmade_result = testing_barlow_twins_loss(z1, z2)
    result = jax.jit(_self_supervised.barlow_twins_loss)(z1, z2)
    np.testing.assert_allclose(result, handmade_result, atol=1e-6)

  def test_raises_on_rank_not_2(self):
    z1 = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    z2 = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      _self_supervised.barlow_twins_loss(z1, z2)

  def test_raises_on_mismatched_shapes(self):
    z1 = jnp.zeros((4, 3), dtype=jnp.float32)
    z2 = jnp.zeros((5, 3), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      _self_supervised.barlow_twins_loss(z1, z2)
if __name__ == '__main__':
  absltest.main()
