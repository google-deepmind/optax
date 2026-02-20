# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for AdaHessian optimizer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from optax import contrib
from optax._src import test_utils


class AdaHessianTest(absltest.TestCase):

  def test_hutchinson_estimator_quadratic_diag(self):
    def obj_fn(params):
      return jnp.sum(params ** 2)

    params = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    estimator = contrib.hutchinson_estimator_diag_hessian(n_samples=4)
    state = estimator.init(params)
    diag, _ = estimator.update(None, state, params=params, obj_fn=obj_fn)

    # For f(x)=sum(x^2), Hessian is 2I, so the diagonal is exactly 2.
    expected = jnp.full_like(params, 2.0)
    test_utils.assert_trees_all_close(diag, expected, rtol=0.0, atol=0.0)

  def test_adahessian_update_runs(self):
    def obj_fn(params):
      return jnp.sum(params ** 2)

    params = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    opt = contrib.adahessian(learning_rate=1e-2, update_interval=2)
    state = opt.init(params)

    grads = jax.grad(obj_fn)(params)
    updates, state = opt.update(grads, state, params, obj_fn=obj_fn)

    def _all_finite(x):
      return jnp.all(jnp.isfinite(x))

    # Ensure the update step yields finite values for both updates and state.
    self.assertTrue(jax.tree.reduce(lambda a, b: a & b,
                                    jax.tree.map(_all_finite, updates)))
    self.assertTrue(jax.tree.reduce(lambda a, b: a & b,
                                    jax.tree.map(_all_finite, state)))


if __name__ == "__main__":
  absltest.main()
