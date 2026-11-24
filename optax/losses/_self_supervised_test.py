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

  def test_small_temperature_tail_precision_and_finite_grad(self):
    embeddings = jnp.array(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 0, 1, 1])
    loss, grad = jax.value_and_grad(_self_supervised.ntxent)(
        embeddings, labels, 0.05
    )
    expected = float(np.log1p(2.0 * np.exp(-40.0)))
    np.testing.assert_allclose(loss, expected, rtol=1e-5)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

  def test_narrow_dtypes_small_temperature(self):
    labels = jnp.array([0, 0, 1, 1])
    loss_f32, grad_f32 = jax.value_and_grad(_self_supervised.ntxent)(
        self.ys, labels, 0.03
    )
    for dtype in (jnp.bfloat16, jnp.float16):
      emb = self.ys.astype(dtype)
      loss, grad = jax.value_and_grad(_self_supervised.ntxent)(
          emb, labels, 0.03
      )
      self.assertEqual(loss.dtype, dtype)
      self.assertEqual(grad.dtype, dtype)
      self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
      np.testing.assert_allclose(
          loss.astype(jnp.float32), loss_f32, rtol=2e-2, atol=2e-2
      )
      np.testing.assert_allclose(
          grad.astype(jnp.float32), grad_f32, rtol=2e-2, atol=5e-2
      )

  def test_all_same_labels_zero_loss_and_grad(self):
    labels = jnp.array([0, 0, 0, 0])
    loss, grad = jax.value_and_grad(_self_supervised.ntxent)(self.ys, labels)
    np.testing.assert_allclose(loss, 0.0, atol=0.0)
    np.testing.assert_allclose(grad, jnp.zeros_like(self.ys), atol=0.0)

  def test_multi_positive_groups_and_vmap(self):
    embeddings = jnp.array([
        [1.0, 0.2],
        [0.9, 0.1],
        [0.8, 0.3],
        [-1.0, -0.2],
        [-0.9, -0.1],
        [0.0, 1.0],
    ])
    labels = jnp.array([0, 0, 0, 1, 1, 2])
    loss, grad = jax.value_and_grad(_self_supervised.ntxent)(embeddings, labels)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    batched_emb = jnp.stack([embeddings, embeddings * 0.5], axis=0)
    batched_labels = jnp.stack([labels, labels], axis=0)
    vmap_losses = jax.jit(jax.vmap(_self_supervised.ntxent, in_axes=(0, 0)))(
        batched_emb, batched_labels
    )
    np.testing.assert_allclose(vmap_losses, jnp.array([loss, loss]), atol=1e-5)


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

  @parameterized.parameters([1, 2, 3])
  def test_norm_degree(self, norm_degree):
    anchor = np.array([[0.0, 0.0], [1.0, 1.0]])
    positive = np.array([[0.1, 0.1], [1.1, 1.1]])
    negative = np.array([[1.0, 0.0], [0.0, 1.0]])
    ap_distance = np.linalg.norm(anchor - positive, ord=norm_degree, axis=-1)
    an_distance = np.linalg.norm(anchor - negative, ord=norm_degree, axis=-1)
    expected = np.maximum(ap_distance - an_distance + 1.0, 0.0)
    result = _self_supervised.triplet_margin_loss(
        anchor, positive, negative, norm_degree=norm_degree, eps=0.0
    )
    np.testing.assert_allclose(result, expected, atol=1e-4)

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


if __name__ == '__main__':
  absltest.main()
