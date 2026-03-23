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
"""Tests for the Sophia optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import update
from optax.contrib import _sophia


def _quadratic(params):
  return jnp.sum(params**2)


class SophiaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = jnp.array([1.0, 2.0, 3.0])
    self.grads = jax.grad(_quadratic)(self.params)

  def test_state_init(self):
    opt = _sophia.sophia(learning_rate=1e-2)
    state = opt.init(self.params)
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_single_step_finite(self):
    opt = _sophia.sophia(learning_rate=1e-2)
    state = opt.init(self.params)
    updates, new_state = opt.update(
        self.grads, state, self.params, obj_fn=_quadratic
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_requires_params(self):
    opt = _sophia.sophia(learning_rate=1e-2)
    state = opt.init(self.params)
    with self.assertRaises(ValueError):
      opt.update(self.grads, state, params=None, obj_fn=_quadratic)

  def test_requires_obj_fn(self):
    opt = _sophia.sophia(learning_rate=1e-2)
    state = opt.init(self.params)
    with self.assertRaises(ValueError):
      opt.update(self.grads, state, self.params)

  @parameterized.parameters(1.0, 2.0, None)
  def test_clip_threshold(self, clip_threshold):
    opt = _sophia.sophia(
        learning_rate=1e-2, clip_threshold=clip_threshold
    )
    state = opt.init(self.params)
    updates, _ = opt.update(
        self.grads, state, self.params, obj_fn=_quadratic
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))

  def test_hessian_update_interval(self):
    """Hessian diagonal should only be updated every update_interval steps."""
    opt = _sophia.sophia(learning_rate=1e-2, update_interval=5)
    state = opt.init(self.params)
    for _ in range(5):
      _, state = opt.update(
          self.grads, state, self.params, obj_fn=_quadratic
      )
    leaves = jax.tree.leaves(state)
    self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

  def test_zero_gradients(self):
    opt = _sophia.sophia(learning_rate=1e-2)
    state = opt.init(self.params)
    zero_grads = jnp.zeros_like(self.params)
    updates, _ = opt.update(
        zero_grads, state, self.params, obj_fn=_quadratic
    )
    self.assertTrue(jnp.all(jnp.isfinite(updates)))


if __name__ == '__main__':
  absltest.main()
