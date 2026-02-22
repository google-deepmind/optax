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
  @parameterized.parameters(0, 1, 42)
  def test_symmetric_batched_equals_avg_of_two_single_direction_calls(
    self, seed):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q1 = jax.random.normal(k1, (2, 3), dtype=jnp.float32)
    q2 = jax.random.normal(k2, (2, 3), dtype=jnp.float32)
    z1 = jax.random.normal(k3, (2, 3), dtype=jnp.float32)
    z2 = jax.random.normal(k4, (2, 3), dtype=jnp.float32)
    byol_jit = jax.jit(_self_supervised.byol_loss)

    symmetric = byol_jit(q1, z2, q2, z1, symmetric=True)
    cos_12 = _regression.cosine_similarity(q1, z2, epsilon=1e-6)
    cos_21 = _regression.cosine_similarity(q2, z1, epsilon=1e-6)
    expected = jnp.mean(0.5 * ((2.0 - 2.0 * cos_12) + (2.0 - 2.0 * cos_21)))
    np.testing.assert_allclose(symmetric, expected, atol=1e-4)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_zero_when_inputs_identical(
    self, seed):
    key = jax.random.key(seed)
    k1 = jax.random.split(key, 2)[0]
    q = jax.random.normal(k1, (2, 3), dtype=jnp.float32)
    z = q

    result = jax.jit(_self_supervised.byol_loss)(q, z, symmetric=False)
    self.assertTrue(np.isfinite(result))
    np.testing.assert_allclose(result, 0.0, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_positive_scaling(
    self, seed):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)
    q = jax.random.normal(k1, (8, 16), dtype=jnp.float32)
    z = jax.random.normal(k2, (8, 16), dtype=jnp.float32)
    byol_jit = jax.jit(_self_supervised.byol_loss)
    base = byol_jit(q, z, symmetric=False)
    scaled = byol_jit(3.0 * q, 0.5 * z, symmetric=False)
    np.testing.assert_allclose(base, scaled, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_batch_permutation(
    self, seed):
    key = jax.random.key(seed)
    k1, k2, kperm = jax.random.split(key, 3)
    q = jax.random.normal(k1, (10, 7), dtype=jnp.float32)
    z = jax.random.normal(k2, (10, 7), dtype=jnp.float32)
    perm = jax.random.permutation(kperm, q.shape[0])

    byol_jit = jax.jit(_self_supervised.byol_loss)
    base = byol_jit(q, z, symmetric=False)
    shuffled = byol_jit(q[perm], z[perm], symmetric=False)
    np.testing.assert_allclose(base, shuffled, atol=1e-5)

class SimSiamLossTest(parameterized.TestCase):
  @parameterized.parameters(0, 1, 42)
  def test_symmetric_equals_avg_of_two_single_direction_calls(
    self, seed):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    p1 = jax.random.normal(k1, (2, 3), dtype=jnp.float32)
    p2 = jax.random.normal(k2, (2, 3), dtype=jnp.float32)
    z1 = jax.random.normal(k3, (2, 3), dtype=jnp.float32)
    z2 = jax.random.normal(k4, (2, 3), dtype=jnp.float32)

    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)
    symmetric = simsiam_jit(p1, z2, p2, z1, symmetric=True)
    cos_12 = _regression.cosine_similarity(p1, z2, epsilon=1e-6)
    cos_21 = _regression.cosine_similarity(p2, z1, epsilon=1e-6)
    expected = jnp.mean(0.5 * ((-cos_12) + (-cos_21)))
    np.testing.assert_allclose(symmetric, expected, atol=1e-4)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_equals_minus_one_when_inputs_identical(
    self, seed):
    key = jax.random.key(seed)
    k1 = jax.random.split(key, 2)[0]
    p = jax.random.normal(k1, (2, 3), dtype=jnp.float32)
    z = p
    result = jax.jit(_self_supervised.simsiam_loss)(p, z, symmetric=False)
    self.assertTrue(np.isfinite(result))
    np.testing.assert_allclose(result, -1.0, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_positive_scaling(
    self, seed):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)
    p = jax.random.normal(k1, (8, 16), dtype=jnp.float32)
    z = jax.random.normal(k2, (8, 16), dtype=jnp.float32)
    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)
    base = simsiam_jit(p, z, symmetric=False)
    scaled = simsiam_jit(2.0 * p, 0.25 * z, symmetric=False)
    np.testing.assert_allclose(base, scaled, atol=1e-5)

class DinoLossTest(parameterized.TestCase):
  @parameterized.parameters(0, 1, 42)
  def test_two_view_matches_handmade(self, seed):
    key = jax.random.key(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    s1 = jax.random.normal(k1, (2, 3), dtype=jnp.float32)
    s2 = jax.random.normal(k2, (2, 3), dtype=jnp.float32)
    t1 = jax.random.normal(k3, (2, 3), dtype=jnp.float32)
    t2 = jax.random.normal(k4, (2, 3), dtype=jnp.float32)
    student_temperature = 0.1
    teacher_temperature = 0.04
    teacher_center = jax.random.normal(k5, (3,), dtype=jnp.float32)

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
        s1, t1,
        student_logits_2=s2,
        teacher_logits_2=t2,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        teacher_center=teacher_center,
        two_view=True,
    )
    np.testing.assert_allclose(result, handmade_result, atol=1e-6)

  @parameterized.parameters(0, 1, 42)
  def test_single_view_invariant_to_logit_translation(self, seed):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)
    student_logits = jax.random.normal(k1, (4, 5), dtype=jnp.float32)
    teacher_logits = jax.random.normal(k2, (4, 5), dtype=jnp.float32)
    student_temperature = 0.1
    teacher_temperature = 0.04
    teacher_center = jnp.zeros((5,), dtype=jnp.float32)

    dino_jit = jax.jit(_self_supervised.dino_loss)
    base = dino_jit(
        student_logits,
        teacher_logits,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        teacher_center=teacher_center,
        two_view=False,
    )
    translated = dino_jit(
        student_logits + 7.0,
        teacher_logits + 7.0,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        teacher_center=teacher_center,
        two_view=False,
    )
    np.testing.assert_allclose(base, translated, atol=1e-6)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_batch_permutation(
    self, seed):
    key = jax.random.key(seed)
    k1, k2, kperm = jax.random.split(key, 3)
    p = jax.random.normal(k1, (10, 7), dtype=jnp.float32)
    z = jax.random.normal(k2, (10, 7), dtype=jnp.float32)
    perm = jax.random.permutation(kperm, p.shape[0])

    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)
    base = simsiam_jit(p, z, symmetric=False)
    shuffled = simsiam_jit(p[perm], z[perm], symmetric=False)
    np.testing.assert_allclose(base, shuffled, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_view_minimized_student_matches_teacher_distribution(
    self, seed):
    key = jax.random.key(seed)
    k = jax.random.split(key, 1)[0]
    teacher_logits = jax.random.normal(k, (4, 6), dtype=jnp.float32)
    teacher_center = jnp.zeros((6,), dtype=jnp.float32)

    temperature = 0.2
    dino_jit = jax.jit(_self_supervised.dino_loss)
    loss_match = dino_jit(
        teacher_logits,  # student logits == teacher logits
        teacher_logits,
        student_temperature=temperature,
        teacher_temperature=temperature,
        teacher_center=teacher_center,
        two_view=False,
    )
    # reverse class dimension (per-example permutation)
    student_mismatch = teacher_logits[:, ::-1]
    loss_mismatch = dino_jit(
        student_mismatch,
        teacher_logits,
        student_temperature=temperature,
        teacher_temperature=temperature,
        teacher_center=teacher_center,
        two_view=False,
    )
    self.assertTrue(np.isfinite(loss_match))
    self.assertTrue(np.isfinite(loss_mismatch))
    self.assertLessEqual(float(loss_match), float(loss_mismatch) + 1e-7)

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

class BarlowTwinsLossTest(parameterized.TestCase):
  # This matrix has per-dim mean is 0, per-dim var is 1,
  #  and columns are orthogonal.
  # This makes the cross-correlation exactly identity when z1 == z2,
  #  so the Barlow loss should be 0.
  def test_zero_for_identity_like_inputs_and_nonzero_otherwise(self):
    z = jnp.array(
        [
            [1.0,  1.0,  1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    loss_same = jax.jit(_self_supervised.barlow_twins_loss)(z, z)
    self.assertTrue(np.isfinite(loss_same))
    np.testing.assert_allclose(loss_same, 0.0, atol=1e-6)

    loss_diff = jax.jit(_self_supervised.barlow_twins_loss)(z, -z)
    self.assertTrue(np.isfinite(loss_diff))
    self.assertGreater(float(loss_diff), 1e-3)

  def test_invariant_to_batch_permutation(self):
    z = jnp.array(
        [
            [1.0,  1.0,  1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    barlow_jit = jax.jit(_self_supervised.barlow_twins_loss)
    base = barlow_jit(z, z)
    shuffled = barlow_jit(z[perm], z[perm])
    np.testing.assert_allclose(base, shuffled, atol=1e-6)

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
