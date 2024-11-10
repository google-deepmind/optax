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

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax.losses import _self_supervised

class NtxentTest(chex.TestCase):

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

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys, self.ts_1),
        self.exp_1,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys, self.ts_2),
        self.exp_2,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys_2, self.ts_1),
        self.exp_3,
        atol=1e-4,
    )

class TripletMarginLossTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.a1 = jnp.ones((2, 2))
    self.p1 = jnp.zeros((2, 2))
    self.n1 = jnp.ones((2, 2))*2
    self.a2 = jnp.zeros((2, 2))
    self.p2 = jnp.ones((2, 2))
    self.n2 = jnp.ones((2, 2))*2

  @chex.all_variants
  def test_batched(self):
    def testing_triplet_loss(a, p, n, margin=1.0, p_norm=2, eps=1e-6):
      ap_distance = jnp.sqrt(jnp.sum(jnp.power(a - p, p_norm)) + eps)
      an_distance = jnp.sqrt(jnp.sum(jnp.power(a - n, p_norm)) + eps)
      return jnp.maximum(ap_distance - an_distance + margin, 0)

    handmade_result = testing_triplet_loss(
    a=self.a1, p=self.p1, n=self.n1, margin=1.0)
    result = self.variant(_self_supervised.triplet_loss)(
    self.a1, self.p1, self.n1)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = testing_triplet_loss(
    a=self.a2, p=self.p2, n=self.n2, margin=1.0)
    result = self.variant(_self_supervised.triplet_loss)(
    self.a2, self.p2, self.n2)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = testing_triplet_loss(
    a=self.a1, p=self.p1, n=self.n1, margin=1.0)
    result = self.variant(_self_supervised.triplet_loss)(
    anchors=self.a1, positives=self.p1, negatives=self.n1)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = testing_triplet_loss(
    a=self.a2, p=self.p2, n=self.n2, margin=1.0)
    result = self.variant(_self_supervised.triplet_loss)(
    anchors=self.a2, positives=self.p2, negatives=self.n2)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

  @chex.all_variants
  def test_vmap(self):
    original_loss = _self_supervised.triplet_loss(self.a1, self.p1, self.n1,
                                                  reduction='none')

    a1_batched = self.a1.reshape(1, *self.a1.shape)
    p1_batched = self.p1.reshape(1, *self.p1.shape)
    n1_batched = self.n1.reshape(1, *self.n1.shape)

    vmap_loss = self.variant(jax.vmap(_self_supervised.triplet_loss,
                                      in_axes=(0, 0, 0)))(a1_batched,
                                      p1_batched, n1_batched)

    np.testing.assert_allclose(vmap_loss.flatten(), original_loss.flatten(),
                               atol=1e-4)


if __name__ == '__main__':
  absltest.main()
