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
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import stochastic_gradient_estimators
from optax._src import utils


# Set seed for deterministic sampling.
np.random.seed(42)


_estimator_to_num_samples = {
    stochastic_gradient_estimators.score_function_jacobians: 5 * 10**5,
    stochastic_gradient_estimators.measure_valued_jacobians: 10**5,
    stochastic_gradient_estimators.pathwise_jacobians: 5 * 10**4,
}

_weighted_estimator_to_num_samples = {
    stochastic_gradient_estimators.score_function_jacobians: 5 * 10**6,
    stochastic_gradient_estimators.measure_valued_jacobians: 5 * 10**5,
    stochastic_gradient_estimators.pathwise_jacobians: 5 * 10**4,
}


def _cross_prod(items1, items2):
  prod = itertools.product(items1, items2)
  return [i1 + (i2,) for i1, i2 in prod]


def _ones(dims):
  return jnp.ones(shape=(dims), dtype=jnp.float32)


def _assert_equal(actual, expected, rtol=1e-2, atol=1e-2):
  # Note: assert_allclose does not check shapes
  assert actual.shape == expected.shape
  # We get around the bug https://github.com/numpy/numpy/issues/13801
  zero_indices = np.argwhere(expected == 0)
  if not np.all(np.abs(actual[zero_indices]) <= atol):
    raise AssertionError('Larger than {} diff in {}'.format(
        atol, actual[zero_indices]))

  non_zero_indices = np.argwhere(expected != 0)
  np.testing.assert_allclose(
      np.asarray(actual)[non_zero_indices],
      expected[non_zero_indices], rtol, atol)


def _estimator_variant(variant, estimator):
  return variant(estimator, static_argnums=(0, 2, 4))


def _measure_valued_variant(variant):
  return variant(
      stochastic_gradient_estimators.measure_valued_jacobians,
      static_argnums=(0, 2, 4, 5))


class GradientEstimatorsTest(chex.TestCase):

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [(0.1,), (0.5,), (0.9,)],
      [
          stochastic_gradient_estimators.score_function_jacobians,
          stochastic_gradient_estimators.pathwise_jacobians,
          stochastic_gradient_estimators.measure_valued_jacobians,
      ],))
  def testConstantFunction(self, constant, estimator):
    data_dims = 3
    num_samples = _estimator_to_num_samples[estimator]

    effective_mean = 1.5
    mean = effective_mean * _ones(data_dims)

    effective_log_scale = 0.0
    log_scale = effective_log_scale * _ones(data_dims)
    rng = jax.random.PRNGKey(1)

    jacobians = _estimator_variant(self.variant, estimator)(
        lambda x: jnp.array(constant), [mean, log_scale],
        utils.multi_normal, rng, num_samples)

    # Average over the number of samples.
    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = np.mean(mean_jacobians, axis=0)
    expected_mean_grads = np.zeros(data_dims, dtype=np.float32)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = np.mean(log_scale_jacobians, axis=0)
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    _assert_equal(mean_grads, expected_mean_grads, atol=5e-3)
    _assert_equal(log_scale_grads, expected_log_scale_grads, atol=5e-3)

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [(0.5, -1.), (0.7, 0.0), (0.8, 0.1)],
      [
          stochastic_gradient_estimators.score_function_jacobians,
          stochastic_gradient_estimators.pathwise_jacobians,
          stochastic_gradient_estimators.measure_valued_jacobians]))
  def testLinearFunction(self, effective_mean, effective_log_scale, estimator):
    data_dims = 3
    num_samples = _estimator_to_num_samples[estimator]
    rng = jax.random.PRNGKey(1)

    mean = effective_mean * _ones(data_dims)
    log_scale = effective_log_scale * _ones(data_dims)

    jacobians = _estimator_variant(self.variant, estimator)(
        np.sum, [mean, log_scale],
        utils.multi_normal, rng, num_samples)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = np.mean(mean_jacobians, axis=0)
    expected_mean_grads = np.ones(data_dims, dtype=np.float32)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = np.mean(log_scale_jacobians, axis=0)
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    _assert_equal(mean_grads, expected_mean_grads)
    _assert_equal(log_scale_grads, expected_log_scale_grads)

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [(1.0, 0.3)],
      [
          stochastic_gradient_estimators.score_function_jacobians,
          stochastic_gradient_estimators.pathwise_jacobians,
          stochastic_gradient_estimators.measure_valued_jacobians,
      ]))
  def testQuadraticFunction(
      self, effective_mean, effective_log_scale, estimator):
    data_dims = 3
    num_samples = _estimator_to_num_samples[estimator]
    rng = jax.random.PRNGKey(1)

    mean = effective_mean * _ones(data_dims)
    log_scale = effective_log_scale * _ones(data_dims)

    jacobians = _estimator_variant(self.variant, estimator)(
        lambda x: np.sum(x**2) / 2, [mean, log_scale],
        utils.multi_normal, rng, num_samples)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = np.mean(mean_jacobians, axis=0)
    expected_mean_grads = effective_mean * np.ones(
        data_dims, dtype=np.float32)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = np.mean(log_scale_jacobians, axis=0)
    expected_log_scale_grads = np.exp(2 * effective_log_scale) * np.ones(
        data_dims, dtype=np.float32)

    _assert_equal(mean_grads, expected_mean_grads, atol=5e-2)
    _assert_equal(log_scale_grads, expected_log_scale_grads, atol=5e-2)

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [([1.0, 2.0, 3.], [-1., 0.3, -2.], [1., 1., 1.]),
       ([1.0, 2.0, 3.], [-1., 0.3, -2.], [4., 2., 3.]),
       ([1.0, 2.0, 3.], [0.1, 0.2, 0.1], [10., 5., 1.])
      ],
      [
          stochastic_gradient_estimators.score_function_jacobians,
          stochastic_gradient_estimators.pathwise_jacobians,
          stochastic_gradient_estimators.measure_valued_jacobians,
      ]))
  def testWeightedLinear(
      self, effective_mean, effective_log_scale, weights, estimator):
    num_samples = _weighted_estimator_to_num_samples[estimator]
    rng = jax.random.PRNGKey(1)

    mean = jnp.array(effective_mean)
    log_scale = jnp.array(effective_log_scale)
    weights = jnp.array(weights)

    data_dims = len(effective_mean)

    function = lambda x: jnp.sum(x* weights)
    jacobians = _estimator_variant(self.variant, estimator)(
        function, [mean, log_scale],
        utils.multi_normal, rng, num_samples)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = np.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = np.mean(log_scale_jacobians, axis=0)

    expected_mean_grads = weights
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    _assert_equal(mean_grads, expected_mean_grads, atol=5e-2)
    _assert_equal(log_scale_grads, expected_log_scale_grads, atol=5e-2)

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [([1.0, 2.0, 3.], [-1., 0.3, -2.], [1., 1., 1.]),
       ([1.0, 2.0, 3.], [-1., 0.3, -2.], [4., 2., 3.]),
       ([1.0, 2.0, 3.], [0.1, 0.2, 0.1], [3., 5., 1.])
      ], [
          stochastic_gradient_estimators.score_function_jacobians,
          stochastic_gradient_estimators.pathwise_jacobians,
          stochastic_gradient_estimators.measure_valued_jacobians,]))
  def testWeightedQuadratic(
      self, effective_mean, effective_log_scale, weights, estimator):
    num_samples = _weighted_estimator_to_num_samples[estimator]
    rng = jax.random.PRNGKey(1)

    mean = jnp.array(effective_mean, dtype=jnp.float32)
    log_scale = jnp.array(effective_log_scale, dtype=jnp.float32)
    weights = jnp.array(weights, dtype=jnp.float32)

    data_dims = len(effective_mean)

    function = lambda x: jnp.sum(x* weights) ** 2
    jacobians = _estimator_variant(self.variant, estimator)(
        function, [mean, log_scale],
        utils.multi_normal, rng, num_samples)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = np.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = np.mean(log_scale_jacobians, axis=0)

    expected_mean_grads = 2 * weights * np.sum(weights * mean)
    effective_scale = np.exp(log_scale)
    expected_scale_grads = 2 * weights ** 2 * effective_scale
    expected_log_scale_grads = expected_scale_grads * effective_scale

    _assert_equal(mean_grads, expected_mean_grads, atol=1e-1, rtol=1e-1)
    _assert_equal(
        log_scale_grads, expected_log_scale_grads, atol=1e-1, rtol=1e-1)

  @chex.all_variants
  @parameterized.parameters(_cross_prod(
      [([1.0], [1.0], lambda x: jnp.sum(jnp.cos(x))),
       # Need to ensure that the mean is not too close to 0.
       ([10.0], [0.0], lambda x: jnp.sum(jnp.log(x))),
       ([1.0, 2.0], [1.0, -2],
        lambda x: jnp.sum(jnp.cos(2 * x))),
       ([1.0, 2.0], [1.0, -2],
        lambda x: jnp.cos(jnp.sum(2 * x))),
      ], [True, False]))
  def testNonPolynomialFunctionConsistencyWithPathwise(
      self, effective_mean, effective_log_scale, function, coupling):
    num_samples = 10**5
    rng = jax.random.PRNGKey(1)
    measure_rng, pathwise_rng = jax.random.split(rng)

    mean = jnp.array(effective_mean, dtype=jnp.float32)
    log_scale = jnp.array(effective_log_scale, dtype=jnp.float32)
    data_dims = len(effective_mean)

    measure_valued_jacobians = _measure_valued_variant(self.variant)(
        function, [mean, log_scale],
        utils.multi_normal, measure_rng, num_samples, coupling)

    measure_valued_mean_jacobians = measure_valued_jacobians[0]
    chex.assert_shape(measure_valued_mean_jacobians, (num_samples, data_dims))
    measure_valued_mean_grads = np.mean(measure_valued_mean_jacobians, axis=0)

    measure_valued_log_scale_jacobians = measure_valued_jacobians[1]
    chex.assert_shape(
        measure_valued_log_scale_jacobians, (num_samples, data_dims))
    measure_valued_log_scale_grads = np.mean(
        measure_valued_log_scale_jacobians, axis=0)

    pathwise_jacobians = _estimator_variant(
        self.variant, stochastic_gradient_estimators.pathwise_jacobians)(
            function, [mean, log_scale],
            utils.multi_normal, pathwise_rng, num_samples)

    pathwise_mean_jacobians = pathwise_jacobians[0]
    chex.assert_shape(pathwise_mean_jacobians, (num_samples, data_dims))
    pathwise_mean_grads = np.mean(pathwise_mean_jacobians, axis=0)

    pathwise_log_scale_jacobians = pathwise_jacobians[1]
    chex.assert_shape(pathwise_log_scale_jacobians, (num_samples, data_dims))
    pathwise_log_scale_grads = np.mean(pathwise_log_scale_jacobians, axis=0)

    _assert_equal(
        pathwise_mean_grads, measure_valued_mean_grads, rtol=5e-1, atol=1e-1)
    _assert_equal(
        pathwise_log_scale_grads, measure_valued_log_scale_grads,
        rtol=5e-1, atol=1e-1)


class MeasuredValuedEstimatorsTest(chex.TestCase):

  @chex.all_variants
  @parameterized.parameters([True, False])
  def testRaisesErrorForNonGaussian(self, coupling):
    num_samples = 10**5
    rng = jax.random.PRNGKey(1)

    function = lambda x: jnp.sum(x) ** 2

    mean = jnp.array(0, dtype=jnp.float32)
    log_scale = jnp.array(0., dtype=jnp.float32)

    class TestDist():

      def __init__(self, params):
        self._params = params

      def sample(self, n):
        return np.zeros(n)

    with self.assertRaises(ValueError):
      _measure_valued_variant(self.variant)(
          function, [mean, log_scale],
          TestDist, rng, num_samples, coupling)


if __name__ == '__main__':
  absltest.main()
