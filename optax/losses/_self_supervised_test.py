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
    self.a_1 = jnp.array([[ 0.8919014 ,  1.43801339],
        [-1.00469918, -0.11962243]])
    self.a_2 = jnp.array([[-0.43486292,  1.23246442],
        [ 0.97871209,  0.09696856]])
    self.p_1 = jnp.array([[-1.76960172,  0.44844366],
        [-0.53867503, -0.18704526]])
    self.p_2 = jnp.array([[ 0.36411669, -0.46389038],
        [ 0.84194035,  0.15887335]])
    self.n_1 = jnp.array( [[-1.1787581 ,  0.10253741],
        [ 0.31388789,  0.75293212]])
    self.n_2 = jnp.array([[ 0.02613382, -0.17087295],
        [ 0.33538733, -0.66962659]])
    self.exp_1 = jnp.array(0.6877749)
    self.exp_2 = jnp.array(0.77367365)

  @chex.all_variants
  def test_batched(self):
    """Tests"""
    np.testing.assert_allclose(
        self.variant(_self_supervised.triplet_margin_loss)(
          self.a_1, self.p_1, self.n_1),
        self.exp_1,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        self.variant(_self_supervised.triplet_margin_loss)(
          self.a_2, self.p_2, self.n_2),
        self.exp_2,
        atol=1e-4,
    )

if __name__ == '__main__':
  absltest.main()
