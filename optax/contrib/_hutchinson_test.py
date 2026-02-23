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
"""Tests for Hutchinson estimator utilities."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from optax import contrib
from optax._src import test_utils


class HutchinsonTest(absltest.TestCase):

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

  def test_hutchinson_estimator_raises_without_required_args(self):
    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    estimator = contrib.hutchinson_estimator_diag_hessian(
        random_seed=jax.random.PRNGKey(0), n_samples=2
    )
    state = estimator.init(params)

    def obj_fn(p):
      return jnp.sum(p["w"] ** 2)

    with self.assertRaisesRegex(ValueError, "params must be provided"):
      estimator.update(None, state, params=None, obj_fn=obj_fn)

    with self.assertRaisesRegex(ValueError, "obj_fn must be provided"):
      estimator.update(None, state, params=params, obj_fn=None)

  def test_hutchinson_estimator_nontrivial_objective_is_finite_and_shaped(self):
    params = {
        "w": jnp.array([0.3, -0.7, 1.1], dtype=jnp.float32),
        "b": jnp.array([0.2], dtype=jnp.float32),
    }

    def obj_fn(p):
      w, b = p["w"], p["b"]
      return jnp.sum(w**4) + jnp.sum(jnp.sin(w + b)) + jnp.sum(b**2)

    estimator = contrib.hutchinson_estimator_diag_hessian(
        random_seed=jax.random.PRNGKey(0), n_samples=8
    )
    state = estimator.init(params)
    diag, _ = estimator.update(None, state, params=params, obj_fn=obj_fn)

    self.assertEqual(
        jax.tree_util.tree_structure(diag), jax.tree_util.tree_structure(params)
    )
    leaves_ok = jax.tree.map(
        lambda d, p: jnp.all(jnp.isfinite(d)) & (d.shape == p.shape), diag, params
    )
    self.assertTrue(jax.tree.reduce(lambda a, b: a & b, leaves_ok))


if __name__ == "__main__":
  absltest.main()
