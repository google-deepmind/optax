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

class ByolLossTest(absltest.TestCase):

  def test_symmetric_batched_matches_handmade(self):
    q1 = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6]],
        dtype=jnp.float32,
    )
    q2 = jnp.array(
        [[0.2, 0.1, 0.0],
         [0.3, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    z1 = jnp.array(
        [[0.5, 0.4, 0.3],
         [0.2, 0.1, 0.0]],
        dtype=jnp.float32,
    )
    z2 = jnp.array(
        [[0.3, 0.2, 0.1],
         [0.6, 0.5, 0.4]],
        dtype=jnp.float32,
    )

    def testing_byol_loss(q1_val, z2_val, q2_val, z1_val, eps=1e-6):

      cos_12 = _regression.cosine_similarity(
          q1_val,
          z2_val,
          epsilon=eps,
      )
      cos_21 = _regression.cosine_similarity(
          q2_val,
          z1_val,
          epsilon=eps,
      )
      loss_12 = 2.0 - 2.0 * cos_12
      loss_21 = 2.0 - 2.0 * cos_21
      loss = 0.5 * (loss_12 + loss_21)
      return jnp.mean(loss)

    handmade_result = testing_byol_loss(q1, z2, q2, z1)
    result = jax.jit(_self_supervised.byol_loss)(
        q1,
        z2,
        q2,
        z1,
        symmetric=True,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

  def test_single_direction_matches_handmade(self):
    q = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    z = jnp.array(
        [[0.4, 0.0, 0.2],
         [0.1, 0.5, 0.3]],
        dtype=jnp.float32,
    )

    def testing_single_direction_byol(q_val, z_val, eps=1e-6):
      cos = _regression.cosine_similarity(
          q_val,
          z_val,
          epsilon=eps,
      )
      loss = 2.0 - 2.0 * cos
      return jnp.mean(loss)

    handmade_result = testing_single_direction_byol(q, z)
    result = jax.jit(_self_supervised.byol_loss)(
        q,
        z,
        symmetric=False,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)


class SimSiamLossTest(absltest.TestCase):

  def test_symmetric_batched_matches_handmade(self):
    p1 = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6]],
        dtype=jnp.float32,
    )
    p2 = jnp.array(
        [[0.2, 0.1, 0.0],
         [0.3, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    z1 = jnp.array(
        [[0.5, 0.4, 0.3],
         [0.2, 0.1, 0.0]],
        dtype=jnp.float32,
    )
    z2 = jnp.array(
        [[0.3, 0.2, 0.1],
         [0.6, 0.5, 0.4]],
        dtype=jnp.float32,
    )

    def testing_simsiam_loss(p1_val, z2_val, p2_val, z1_val, eps=1e-6):
      cos_12 = _regression.cosine_similarity(
          p1_val,
          z2_val,
          epsilon=eps,
      )
      cos_21 = _regression.cosine_similarity(
          p2_val,
          z1_val,
          epsilon=eps,
      )
      loss_12 = -cos_12
      loss_21 = -cos_21
      loss = 0.5 * (loss_12 + loss_21)
      return jnp.mean(loss)

    handmade_result = testing_simsiam_loss(p1, z2, p2, z1)
    result = jax.jit(_self_supervised.simsiam_loss)(
        p1,
        z2,
        p2,
        z1,
        symmetric=True,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

  def test_single_direction_matches_handmade(self):
    p = jnp.array(
        [[0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    z = jnp.array(
        [[0.4, 0.0, 0.2],
         [0.1, 0.5, 0.3]],
        dtype=jnp.float32,
    )

    def testing_single_direction_simsiam(p_val, z_val, eps=1e-6):
      cos = _regression.cosine_similarity(
          p_val,
          z_val,
          epsilon=eps,
      )
      loss = -cos
      return jnp.mean(loss)

    handmade_result = testing_single_direction_simsiam(p, z)
    result = jax.jit(_self_supervised.simsiam_loss)(
        p,
        z,
        symmetric=False,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)


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
