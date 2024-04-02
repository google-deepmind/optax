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
"""Tests for optax.losses._classification."""

from absl.testing import parameterized

import chex
import jax.numpy as jnp
import numpy as np

from optax.losses import _contrastive


class NtxentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([
    [-1.9540, 1.0780],
    [ 0.2380, -0.5703],
    [ 1.8745, -0.0195],
    [-0.6719, -1.9210],
    ])
    self.ts = jnp.array([0,0,1,1])
    # Calculated expected output
    self.exp = jnp.array(14.01032)

  @chex.all_variants
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_contrastive.ntxent)(self.ys, self.ts),
        self.exp, atol=1e-4)
