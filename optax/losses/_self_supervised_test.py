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
"""Tests for optax.losses._self_supervised."""

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
    self.ts_1 = jnp.array([0, 0, 1, 1])
    self.ts_2 = jnp.array([0, 0, 0, 1])
    # Calculated expected output
    self.exp_1 = jnp.array(14.01032)
    self.exp_2 = jnp.array(8.968544)

  @chex.all_variants
  def test_batched(self):
    """Tests for a full batch."""
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


class TripletMarginLossTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.t = jnp.random.normal((2,2))
    self.a1 = self.t*0
    self.p1 = self.a1+1
    self.n1 = self.p1+1

    self.a2 = self.t*0+1
    self.p2 = self.a2-1
    self.n2 = self.a2+1

  def testing_triplet_loss(self, a, p, n, margin=1.0, swap=False):
    ap = jnp.linalg.norm(a - p)
    an = jnp.linalg.norm(a - n)
    if swap:
      pn = jnp.linalg.norm(p - n)
      an = min(an, pn)
    return jnp.maximum(ap - an + margin, 0)
  @chex.all_variants
  def test_batched(self):
    handmade_result = self.variant(self.testing_triplet_loss)(
    self.a1, self.p1, self.n1)
    result = self.variant(_self_supervised.triplet_margin_loss)(
    self.a1, self.p1, self.n1)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = self.variant(self.testing_triplet_loss)(
    self.a2, self.p2, self.n2)
    result = self.variant(_self_supervised.triplet_margin_loss)(
    self.a2, self.p2, self.n2)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = self.variant(self.testing_triplet_loss)(
    self.a1, self.p1, self.n1, swap=True)
    result = self.variant(_self_supervised.triplet_margin_loss)(
    self.a1, self.p1, self.n1, swap=True)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

    handmade_result = self.variant(self.testing_triplet_loss)(
    self.a2, self.p2, self.n2, swap=True)
    result = self.variant(_self_supervised.triplet_margin_loss)(
    self.a2, self.p2, self.n2, swap=True)
    np.testing.assert_allclose(result, handmade_result, atol=1e-4)

  @chex.all_variants
  def test_jit_vmap_compatibility(self):
    # Original function result
    original_loss = _self_supervised.triplet_margin_loss(
                    self.a1, self.p1, self.n1)

    # JIT compiled function result
    jit_loss = (self.variant(jax.jit(
               _self_supervised.triplet_margin_loss))
               (self.a1, self.p1, self.n1))
    np.testing.assert_allclose(jit_loss, original_loss,
                               atol=1e-4)

    # VMAP applied function result
    vmap_loss = self.variant(jax.vmap(
                _self_supervised.triplet_margin_loss,
                in_axes=(0, 0, 0)))(self.a1, self.p1,
                self.n1)
    np.testing.assert_allclose(vmap_loss, original_loss, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
