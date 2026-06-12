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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import utils
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
  def test_single_direction_zero_for_identical_inputs(self, seed):
    key = jax.random.key(seed)
    projections = jax.random.normal(key, (8, 16), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.byol_loss)(projections, projections)

    self.assertEqual(loss.shape, (8,))
    self.assertTrue(np.all(np.isfinite(loss)))
    np.testing.assert_allclose(loss, jnp.zeros(8), atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_nonzero_for_opposite_inputs(self, seed):
    key = jax.random.key(seed)
    projections = jax.random.normal(key, (8, 16), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.byol_loss)(projections, -projections)

    self.assertTrue(np.all(np.isfinite(loss)))
    self.assertGreater(float(loss.min()), 1.0)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_positive_scaling(self, seed):
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    online = jax.random.normal(key1, (8, 16), dtype=jnp.float32)
    target = jax.random.normal(key2, (8, 16), dtype=jnp.float32)
    byol_jit = jax.jit(_self_supervised.byol_loss)

    base = byol_jit(online, target)
    scaled = byol_jit(3.0 * online, 0.5 * target)

    np.testing.assert_allclose(base, scaled, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_two_view_equals_average_of_single_direction_calls(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    online_1 = jax.random.normal(key1, (8, 16), dtype=jnp.float32)
    target_2 = jax.random.normal(key2, (8, 16), dtype=jnp.float32)
    online_2 = jax.random.normal(key3, (8, 16), dtype=jnp.float32)
    target_1 = jax.random.normal(key4, (8, 16), dtype=jnp.float32)
    byol_jit = jax.jit(_self_supervised.byol_loss)

    two_view = byol_jit(online_1, target_2, online_2, target_1)
    expected = 0.5 * (
        byol_jit(online_1, target_2) + byol_jit(online_2, target_1)
    )

    np.testing.assert_allclose(two_view, expected, atol=1e-6)

  @parameterized.parameters(0, 1, 42)
  def test_equivariant_to_batch_permutation(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    online = jax.random.normal(key1, (10, 7), dtype=jnp.float32)
    target = jax.random.normal(key2, (10, 7), dtype=jnp.float32)
    permutation = jax.random.permutation(key3, online.shape[0])
    byol_jit = jax.jit(_self_supervised.byol_loss)

    base = byol_jit(online, target)
    shuffled = byol_jit(online[permutation], target[permutation])

    np.testing.assert_allclose(shuffled, base[permutation], atol=1e-5)

  def test_stops_gradient_through_target_projection(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    online = jax.random.normal(key1, (4, 5), dtype=jnp.float32)
    target = jax.random.normal(key2, (4, 5), dtype=jnp.float32)

    grad_online, grad_target = jax.grad(
        lambda o, t: _self_supervised.byol_loss(o, t).mean(), argnums=(0, 1)
    )(online, target)

    self.assertGreater(float(jnp.linalg.norm(grad_online)), 0.0)
    np.testing.assert_allclose(grad_target, jnp.zeros_like(target))

  def test_finite_value_and_gradient_for_zero_norm_inputs(self):
    projections = jnp.zeros((4, 5), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.byol_loss)(projections, projections)
    grad_fn = jax.grad(
        lambda o: _self_supervised.byol_loss(o, projections).sum()
    )
    grads = grad_fn(projections)

    self.assertTrue(np.all(np.isfinite(loss)))
    self.assertTrue(np.all(np.isfinite(grads)))

  def test_raises_for_incomplete_or_mismatched_two_view_inputs(self):
    projections = jnp.ones((4, 5), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      _self_supervised.byol_loss(
          projections, projections, online_prediction_2=projections
      )
    with self.assertRaises(ValueError):
      _self_supervised.byol_loss(
          projections, projections, projections[:3], projections[:3]
      )

  def test_vmap(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    online = jax.random.normal(key1, (3, 8, 16), dtype=jnp.float32)
    target = jax.random.normal(key2, (3, 8, 16), dtype=jnp.float32)

    vmapped = jax.jit(jax.vmap(_self_supervised.byol_loss))(online, target)
    expected = jnp.stack([
        _self_supervised.byol_loss(online[i], target[i])
        for i in range(online.shape[0])
    ])

    np.testing.assert_allclose(vmapped, expected, atol=1e-6)


class SimSiamLossTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_minus_one_for_identical_inputs(self, seed):
    key = jax.random.key(seed)
    projections = jax.random.normal(key, (8, 16), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.simsiam_loss)(projections, projections)

    self.assertEqual(loss.shape, (8,))
    self.assertTrue(np.all(np.isfinite(loss)))
    np.testing.assert_allclose(loss, -jnp.ones(8), atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_single_direction_invariant_to_positive_scaling(self, seed):
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    prediction = jax.random.normal(key1, (8, 16), dtype=jnp.float32)
    target = jax.random.normal(key2, (8, 16), dtype=jnp.float32)
    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)

    base = simsiam_jit(prediction, target)
    scaled = simsiam_jit(2.0 * prediction, 0.25 * target)

    np.testing.assert_allclose(base, scaled, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_two_view_equals_average_of_single_direction_calls(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    prediction_1 = jax.random.normal(key1, (8, 16), dtype=jnp.float32)
    target_2 = jax.random.normal(key2, (8, 16), dtype=jnp.float32)
    prediction_2 = jax.random.normal(key3, (8, 16), dtype=jnp.float32)
    target_1 = jax.random.normal(key4, (8, 16), dtype=jnp.float32)
    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)

    two_view = simsiam_jit(prediction_1, target_2, prediction_2, target_1)
    expected = 0.5 * (
        simsiam_jit(prediction_1, target_2)
        + simsiam_jit(prediction_2, target_1)
    )

    np.testing.assert_allclose(two_view, expected, atol=1e-6)

  @parameterized.parameters(0, 1, 42)
  def test_equivariant_to_batch_permutation(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    prediction = jax.random.normal(key1, (10, 7), dtype=jnp.float32)
    target = jax.random.normal(key2, (10, 7), dtype=jnp.float32)
    permutation = jax.random.permutation(key3, prediction.shape[0])
    simsiam_jit = jax.jit(_self_supervised.simsiam_loss)

    base = simsiam_jit(prediction, target)
    shuffled = simsiam_jit(prediction[permutation], target[permutation])

    np.testing.assert_allclose(shuffled, base[permutation], atol=1e-5)

  def test_stops_gradient_through_target_projection(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    prediction = jax.random.normal(key1, (4, 5), dtype=jnp.float32)
    target = jax.random.normal(key2, (4, 5), dtype=jnp.float32)

    grad_prediction, grad_target = jax.grad(
        lambda p, t: _self_supervised.simsiam_loss(p, t).mean(), argnums=(0, 1)
    )(prediction, target)

    self.assertGreater(float(jnp.linalg.norm(grad_prediction)), 0.0)
    np.testing.assert_allclose(grad_target, jnp.zeros_like(target))

  def test_finite_value_and_gradient_for_zero_norm_inputs(self):
    projections = jnp.zeros((4, 5), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.simsiam_loss)(projections, projections)
    grad_fn = jax.grad(
        lambda p: _self_supervised.simsiam_loss(p, projections).sum()
    )
    grads = grad_fn(projections)

    self.assertTrue(np.all(np.isfinite(loss)))
    self.assertTrue(np.all(np.isfinite(grads)))

  def test_raises_for_incomplete_or_mismatched_two_view_inputs(self):
    projections = jnp.ones((4, 5), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      _self_supervised.simsiam_loss(
          projections, projections, prediction_2=projections
      )
    with self.assertRaises(ValueError):
      _self_supervised.simsiam_loss(
          projections, projections, projections[:3], projections[:3]
      )

  def test_vmap(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    prediction = jax.random.normal(key1, (3, 8, 16), dtype=jnp.float32)
    target = jax.random.normal(key2, (3, 8, 16), dtype=jnp.float32)

    vmapped = jax.jit(jax.vmap(_self_supervised.simsiam_loss))(
        prediction, target
    )
    expected = jnp.stack([
        _self_supervised.simsiam_loss(prediction[i], target[i])
        for i in range(prediction.shape[0])
    ])

    np.testing.assert_allclose(vmapped, expected, atol=1e-6)


class DinoLossTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 42)
  def test_single_pair_invariant_to_logit_translation(self, seed):
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    student_logits = jax.random.normal(key1, (4, 6), dtype=jnp.float32)
    teacher_logits = jax.random.normal(key2, (4, 6), dtype=jnp.float32)
    dino_jit = jax.jit(_self_supervised.dino_loss)

    base = dino_jit(student_logits, teacher_logits)
    translated = dino_jit(student_logits + 7.0, teacher_logits + 7.0)

    self.assertEqual(base.shape, (4,))
    np.testing.assert_allclose(base, translated, atol=1e-5)

  @parameterized.parameters(0, 1, 42)
  def test_two_view_equals_average_of_cross_view_single_calls(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    student_1 = jax.random.normal(key1, (4, 6), dtype=jnp.float32)
    teacher_1 = jax.random.normal(key2, (4, 6), dtype=jnp.float32)
    student_2 = jax.random.normal(key3, (4, 6), dtype=jnp.float32)
    teacher_2 = jax.random.normal(key4, (4, 6), dtype=jnp.float32)
    center = jax.random.normal(key5, (6,), dtype=jnp.float32)
    dino_jit = jax.jit(_self_supervised.dino_loss)

    two_view = dino_jit(
        student_1, teacher_2, student_2, teacher_1, teacher_center=center
    )
    expected = 0.5 * (
        dino_jit(student_1, teacher_2, teacher_center=center)
        + dino_jit(student_2, teacher_1, teacher_center=center)
    )

    np.testing.assert_allclose(two_view, expected, atol=1e-6)

  @parameterized.parameters(0, 1, 42)
  def test_matching_distribution_not_worse_than_mismatch(self, seed):
    key = jax.random.key(seed)
    teacher_logits = jax.random.normal(key, (4, 6), dtype=jnp.float32)
    dino_jit = jax.jit(_self_supervised.dino_loss)

    loss_match = dino_jit(
        teacher_logits,
        teacher_logits,
        student_temperature=0.2,
        teacher_temperature=0.2,
    )
    loss_mismatch = dino_jit(
        teacher_logits[:, ::-1],
        teacher_logits,
        student_temperature=0.2,
        teacher_temperature=0.2,
    )

    self.assertTrue(np.all(np.isfinite(loss_match)))
    self.assertTrue(np.all(np.isfinite(loss_mismatch)))
    self.assertLessEqual(
        float(loss_match.mean()), float(loss_mismatch.mean()) + 1e-6
    )

  def test_stops_gradient_through_teacher_logits(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    student_logits = jax.random.normal(key1, (4, 6), dtype=jnp.float32)
    teacher_logits = jax.random.normal(key2, (4, 6), dtype=jnp.float32)

    grad_student, grad_teacher = jax.grad(
        lambda s, t: _self_supervised.dino_loss(s, t).mean(), argnums=(0, 1)
    )(student_logits, teacher_logits)

    self.assertGreater(float(jnp.linalg.norm(grad_student)), 0.0)
    np.testing.assert_allclose(grad_teacher, jnp.zeros_like(teacher_logits))

  def test_raises_for_invalid_arguments(self):
    student_logits = jnp.zeros((4, 6), dtype=jnp.float32)
    teacher_logits = jnp.zeros((4, 6), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(student_logits, teacher_logits[:3])
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits, teacher_logits, student_logits_2=student_logits
      )
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          student_logits[:3],
          teacher_logits[:3],
      )
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits, teacher_logits, student_temperature=0.0
      )
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          student_temperature=jnp.ones((2,), dtype=jnp.float32),
      )
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits, teacher_logits, teacher_temperature=-0.1
      )
    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          teacher_center=jnp.zeros((5,), dtype=jnp.float32),
      )
    with self.assertRaises(ValueError):
      # Broadcasts with, but not to, the teacher logits shape.
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          teacher_center=jnp.zeros((2, 1, 1), dtype=jnp.float32),
      )
    with self.assertRaises(TypeError):
      _self_supervised.dino_loss(
          student_logits,
          teacher_logits,
          teacher_center=jnp.zeros((6,), dtype=jnp.int32),
      )

  def test_temperature_validation_is_skipped_for_traced_values(self):
    student_logits = jnp.zeros((4, 6), dtype=jnp.float32)
    teacher_logits = jnp.zeros((4, 6), dtype=jnp.float32)

    with self.assertRaises(ValueError):
      _self_supervised.dino_loss(
          student_logits, teacher_logits, student_temperature=-0.1
      )

    # Traced temperatures cannot be validated at trace time, so the same
    # value compiles and runs when passed as a traced array.
    def loss_fn(temperature):
      return _self_supervised.dino_loss(
          student_logits, teacher_logits, student_temperature=temperature
      )

    traced = jax.jit(loss_fn)(jnp.asarray(-0.1, dtype=jnp.float32))
    self.assertEqual(traced.shape, (4,))

  def test_temperatures_use_logit_dtype_under_x64(self):
    student_logits = jnp.zeros((4, 6), dtype=jnp.float32)
    teacher_logits = jnp.zeros((4, 6), dtype=jnp.float32)

    with utils.x64_precision(True):
      loss = jax.jit(_self_supervised.dino_loss)(
          student_logits,
          teacher_logits,
          student_temperature=0.1,
          teacher_temperature=0.04,
      )

    self.assertEqual(loss.dtype, jnp.float32)

  def test_vmap(self):
    key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    student_logits = jax.random.normal(key1, (3, 4, 6), dtype=jnp.float32)
    teacher_logits = jax.random.normal(key2, (3, 4, 6), dtype=jnp.float32)
    center = jax.random.normal(key3, (6,), dtype=jnp.float32)

    dino = functools.partial(_self_supervised.dino_loss, teacher_center=center)
    vmapped = jax.jit(jax.vmap(dino))(student_logits, teacher_logits)
    expected = jnp.stack([
        dino(student_logits[i], teacher_logits[i])
        for i in range(student_logits.shape[0])
    ])

    np.testing.assert_allclose(vmapped, expected, atol=1e-6)


def _random_decorrelated_projections(key, batch_size, feature_dim):
  """Returns random projections whose cross-correlation matrix is identity.

  Centering a random matrix makes every column orthogonal to the all-ones
  vector, so the Q factor of its QR decomposition has orthonormal, zero-mean
  columns; scaling by `sqrt(batch_size)` gives unit-variance, decorrelated
  features.
  """
  projections = jax.random.normal(
      key, (batch_size, feature_dim), dtype=jnp.float32
  )
  projections = projections - projections.mean(axis=0, keepdims=True)
  q, _ = jnp.linalg.qr(projections)
  return q * jnp.sqrt(batch_size)


class BarlowTwinsLossTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 42)
  def test_zero_for_decorrelated_inputs_and_nonzero_otherwise(self, seed):
    projections = _random_decorrelated_projections(
        jax.random.key(seed), batch_size=8, feature_dim=5
    )

    loss_same = jax.jit(_self_supervised.barlow_twins_loss)(
        projections, projections
    )
    loss_opposite = jax.jit(_self_supervised.barlow_twins_loss)(
        projections, -projections
    )

    self.assertTrue(np.isfinite(loss_same))
    self.assertTrue(np.isfinite(loss_opposite))
    np.testing.assert_allclose(loss_same, 0.0, atol=1e-6)
    self.assertGreater(float(loss_opposite), 1.0)

  @parameterized.parameters(0, 1, 42)
  def test_invariant_to_batch_permutation(self, seed):
    key = jax.random.key(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    projection_1 = jax.random.normal(key1, (10, 7), dtype=jnp.float32)
    projection_2 = jax.random.normal(key2, (10, 7), dtype=jnp.float32)
    permutation = jax.random.permutation(key3, projection_1.shape[0])
    barlow_jit = jax.jit(_self_supervised.barlow_twins_loss)

    base = barlow_jit(projection_1, projection_2)
    shuffled = barlow_jit(
        projection_1[permutation], projection_2[permutation]
    )

    np.testing.assert_allclose(base, shuffled, atol=1e-5)

  def test_constant_inputs_are_finite(self):
    projections = jnp.ones((4, 3), dtype=jnp.float32)

    loss = jax.jit(_self_supervised.barlow_twins_loss)(
        projections, projections
    )

    self.assertTrue(np.isfinite(loss))

  @parameterized.parameters(5e-3, 0.5)
  def test_off_diagonal_term_scales_with_off_diagonal_scale(
      self, off_diagonal_scale
  ):
    feature = jax.random.normal(jax.random.key(0), (8,), dtype=jnp.float32)
    # Duplicating one feature makes the standardized cross-correlation matrix
    # all ones, so the on-diagonal loss is ~0 and the off-diagonal sum of
    # squares is 2; the loss must equal off_diagonal_scale * 2.
    projections = jnp.stack([feature, feature], axis=1)

    loss = jax.jit(
        functools.partial(
            _self_supervised.barlow_twins_loss,
            off_diagonal_scale=off_diagonal_scale,
        )
    )(projections, projections)

    np.testing.assert_allclose(loss, 2.0 * off_diagonal_scale, rtol=1e-3)

  def test_raises_for_invalid_shapes(self):
    with self.assertRaises(ValueError):
      _self_supervised.barlow_twins_loss(
          jnp.zeros((4, 3), dtype=jnp.float32),
          jnp.zeros((5, 3), dtype=jnp.float32),
      )
    with self.assertRaises(ValueError):
      _self_supervised.barlow_twins_loss(
          jnp.zeros((2, 3, 4), dtype=jnp.float32),
          jnp.zeros((2, 3, 4), dtype=jnp.float32),
      )
    with self.assertRaises(ValueError):
      _self_supervised.barlow_twins_loss(
          jnp.zeros((0, 3), dtype=jnp.float32),
          jnp.zeros((0, 3), dtype=jnp.float32),
      )
    with self.assertRaises(ValueError):
      # Per-feature statistics require at least two examples.
      _self_supervised.barlow_twins_loss(
          jnp.zeros((1, 3), dtype=jnp.float32),
          jnp.zeros((1, 3), dtype=jnp.float32),
      )

  def test_vmap(self):
    key1, key2 = jax.random.split(jax.random.key(0))
    projection_1 = jax.random.normal(key1, (3, 8, 5), dtype=jnp.float32)
    projection_2 = jax.random.normal(key2, (3, 8, 5), dtype=jnp.float32)

    vmapped = jax.jit(jax.vmap(_self_supervised.barlow_twins_loss))(
        projection_1, projection_2
    )
    expected = jnp.stack([
        _self_supervised.barlow_twins_loss(projection_1[i], projection_2[i])
        for i in range(projection_1.shape[0])
    ])

    np.testing.assert_allclose(vmapped, expected, atol=1e-6)


class SelfSupervisedDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('byol', _self_supervised.byol_loss),
      ('simsiam', _self_supervised.simsiam_loss),
      ('dino', _self_supervised.dino_loss),
      ('barlow_twins', _self_supervised.barlow_twins_loss),
  )
  def test_raises_for_non_float_inputs(self, loss_fn):
    inputs = jnp.ones((4, 3), dtype=jnp.int32)
    with self.assertRaises(TypeError):
      loss_fn(inputs, inputs)

  def test_byol_raises_for_non_float_second_pair(self):
    projections = jnp.ones((4, 3), dtype=jnp.float32)
    bad = jnp.ones((4, 3), dtype=jnp.int32)
    with self.assertRaises(TypeError):
      _self_supervised.byol_loss(projections, projections, bad, bad)


if __name__ == '__main__':
  absltest.main()
