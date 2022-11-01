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
"""Tests for `second_order.py`."""

import collections
import functools
import itertools

from absl.testing import absltest

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import second_order


NUM_CLASSES = 2
NUM_SAMPLES = 3
NUM_FEATURES = 4


class SecondOrderTest(chex.TestCase):

  def setUp(self):
    super().setUp()

    self.data = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
    self.labels = np.random.randint(NUM_CLASSES, size=NUM_SAMPLES)

    def net_fn(z):
      mlp = hk.Sequential(
          [hk.Linear(10), jax.nn.relu, hk.Linear(NUM_CLASSES)], name='mlp')
      return jax.nn.log_softmax(mlp(z))

    net = hk.without_apply_rng(hk.transform(net_fn))
    self.parameters = net.init(jax.random.PRNGKey(0), self.data)

    def loss(params, inputs, targets):
      log_probs = net.apply(params, inputs)
      return -jnp.mean(hk.one_hot(targets, NUM_CLASSES) * log_probs)

    self.loss_fn = loss

    def jax_hessian_diag(loss_fun, params, inputs, targets):
      """This is the 'ground-truth' obtained via the JAX library."""
      hess = jax.hessian(loss_fun)(params, inputs, targets)

      # Extracts the diagonal components.
      hess_diag = collections.defaultdict(dict)
      for k0, k1 in itertools.product(params.keys(), ['w', 'b']):
        params_shape = params[k0][k1].shape
        n_params = np.prod(params_shape)
        hess_diag[k0][k1] = jnp.diag(hess[k0][k1][k0][k1].reshape(
            n_params, n_params)).reshape(params_shape)
      for k, v in hess_diag.items():
        hess_diag[k] = v
      return second_order.ravel(hess_diag)

    self.hessian = jax_hessian_diag(
        self.loss_fn, self.parameters, self.data, self.labels)

  @chex.all_variants
  def test_hessian_diag(self):
    hessian_diag_fn = self.variant(
        functools.partial(second_order.hessian_diag, self.loss_fn))
    actual = hessian_diag_fn(self.parameters, self.data, self.labels)
    np.testing.assert_array_almost_equal(self.hessian, actual, 5)

  @chex.all_variants
  def test_fisher_diag_shape(self):
    fisher_diag_fn = self.variant(
        functools.partial(second_order.fisher_diag, self.loss_fn))
    fisher_diagonal = fisher_diag_fn(self.parameters, self.data, self.labels)
    chex.assert_equal_shape([fisher_diagonal, self.hessian])


if __name__ == '__main__':
  absltest.main()
