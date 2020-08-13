# Lint as: python3
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
"""Test utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from optax._src import utils
import scipy


# Set seed for deterministic sampling.
np.random.seed(42)


class TestRademacher(chex.TestCase):

  @chex.all_variants()
  def testValues(self):
    rng = jax.random.PRNGKey(1)
    num_samples = 10**5
    values = self.variant(utils.rademacher, static_argnums=1)(
        rng, (num_samples,))
    unique_values, counts = np.unique(values, return_counts=True)
    assert len(unique_values) == 2
    assert len(counts) == 2

    chex.assert_tree_all_close(
        counts[0]/ num_samples, 0.5, rtol=1e-03, atol=1e-03)
    chex.assert_tree_all_close(
        counts[1]/ num_samples, 0.5, rtol=1e-03, atol=1e-03)


class TestDoubleSidedMaxwell(chex.TestCase):

  @chex.all_variants()
  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testDoublesidedMaxwellSample(self, loc, scale):
    num_samples = 10**5
    rng = jax.random.PRNGKey(1)

    samples = self.variant(utils.double_sided_maxwell, static_argnums=3)(
        rng, loc, scale, (num_samples,))

    mean = loc
    std = np.sqrt(3.) * scale

    # Check first and second moments.
    self.assertEqual((num_samples,), samples.shape)
    chex.assert_tree_all_close(np.mean(samples), mean, atol=0., rtol=0.1)
    chex.assert_tree_all_close(np.std(samples), std, atol=0., rtol=0.1)


class TestWeibull(chex.TestCase):

  @chex.all_variants()
  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testSample(self, concentration, scale):
    num_samples = 10**5
    rng = jax.random.PRNGKey(1)

    samples = self.variant(utils.weibull_min, static_argnums=3)(
        rng, scale, concentration, (num_samples,))

    loc = scipy.stats.weibull_min.mean(c=concentration, scale=scale)
    std = scipy.stats.weibull_min.std(c=concentration, scale=scale)
    # Check first and second moments.
    self.assertEqual((num_samples,), samples.shape)
    chex.assert_tree_all_close(np.mean(samples), loc, atol=0., rtol=0.1)
    chex.assert_tree_all_close(np.std(samples), std, atol=0., rtol=0.1)


class TestOneSidedMaxwell(chex.TestCase):

  @chex.all_variants()
  def testSample(self):
    num_samples = 10**5
    rng = jax.random.PRNGKey(1)

    samples = self.variant(utils.one_sided_maxwell, static_argnums=1)(
        rng, (num_samples,))

    loc = scipy.stats.maxwell.mean()
    std = scipy.stats.maxwell.std()
    # Check first and second moments.
    self.assertEqual((num_samples,), samples.shape)
    chex.assert_tree_all_close(np.mean(samples), loc, atol=0., rtol=0.1)
    chex.assert_tree_all_close(np.std(samples), std, atol=0., rtol=0.1)

if __name__ == '__main__':
  absltest.main()
