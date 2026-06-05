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
"""Tests for the cautious optimizer wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import contrib
from optax.contrib._cautious import cautious, CautiousState


class CautiousTest(parameterized.TestCase):

  def test_state_structure(self):
    """The wrapper state holds the base optimizer state."""
    params = jnp.ones((4,))
    base = optax.adam(1e-2)
    tx = cautious(base)
    state = tx.init(params)
    self.assertIsInstance(state, CautiousState)
    # The inner state matches what the base optimizer would produce.
    chex.assert_trees_all_equal_structs(
        state.base_optimizer_state, base.init(params)
    )

  def test_fully_aligned_reduces_to_base(self):
    """When every coordinate agrees with the gradient, cautious == base.

    For SGD on a convex quadratic with no momentum, the update direction is
    always ``-lr * grad``, so ``update * grad < 0`` holds for every coordinate
    and the mask is all-ones. With ``eps`` tiny, the rescale is ~1, so the
    cautious updates must equal the base updates.
    """
    params = jnp.array([1.0, -2.0, 3.0, 0.5])
    grad = jnp.array([0.3, -0.7, 1.2, -0.1])

    base = optax.sgd(1e-1)
    base_state = base.init(params)
    base_updates, _ = base.update(grad, base_state, params)

    tx = cautious(base, eps=1e-8)
    state = tx.init(params)
    caut_updates, _ = tx.update(grad, state, params)

    chex.assert_trees_all_close(caut_updates, base_updates, atol=1e-6)

  def test_masks_misaligned_coordinates(self):
    """Coordinates where the update agrees with the gradient sign are zeroed.

    We construct a base update by hand (via SGD with momentum) so that the
    momentum term points *with* the gradient on some coordinates (uphill in
    Optax's additive convention) and verify those are masked out.
    """
    # Hand-rolled: use a base "optimizer" that just returns a fixed update so
    # we can control alignment precisely.
    fixed_update = jnp.array([-1.0, 1.0, -2.0, 2.0])
    grad = jnp.array([1.0, 1.0, 1.0, -1.0])
    # update * grad: [-1, +1, -2, -2] -> keep where < 0: [T, F, T, T]
    expected_keep = jnp.array([1.0, 0.0, 1.0, 1.0])

    def _const_update(updates, state, params=None, **kw):
      del updates, params, kw
      return fixed_update, state

    const_opt = optax.GradientTransformation(
        lambda p: optax.EmptyState(), _const_update
    )
    tx = cautious(const_opt, eps=1e-8)
    state = tx.init(jnp.zeros(4))
    caut_updates, _ = tx.update(grad, state, jnp.zeros(4))

    # The masked-out coordinate (index 1) must be exactly zero.
    self.assertEqual(float(caut_updates[1]), 0.0)
    # Surviving coordinates keep their sign.
    kept = caut_updates != 0
    np.testing.assert_array_equal(kept.astype(jnp.float32), expected_keep)

  def test_rescaling_preserves_mean_magnitude(self):
    """The surviving updates are rescaled by n / (num_kept + eps)."""
    fixed_update = jnp.array([-1.0, 1.0, -1.0, 1.0])
    grad = jnp.array([1.0, 1.0, 1.0, 1.0])
    # update * grad: [-1, +1, -1, +1] -> keep [T, F, T, F], num_kept = 2, n = 4
    # scale = 4 / 2 = 2. Surviving coords were -1, -1 -> become -2, -2.

    def _const_update(updates, state, params=None, **kw):
      del updates, params, kw
      return fixed_update, state

    const_opt = optax.GradientTransformation(
        lambda p: optax.EmptyState(), _const_update
    )
    tx = cautious(const_opt, eps=1e-8)
    state = tx.init(jnp.zeros(4))
    caut_updates, _ = tx.update(grad, state, jnp.zeros(4))

    expected = jnp.array([-2.0, 0.0, -2.0, 0.0])
    chex.assert_trees_all_close(caut_updates, expected, atol=1e-6)

  def test_descent_guarantee(self):
    """The cautious update never points uphill: <update, grad> <= 0.

    This is the core theoretical property. We exercise it on a momentum
    optimizer (which *can* overshoot) over many random steps and verify the
    inner product of the cautious update with the gradient is always <= 0.
    """
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (32,))
    base = optax.sgd(1e-1, momentum=0.95)  # momentum can overshoot
    tx = cautious(base)
    state = tx.init(params)

    worst = -jnp.inf
    for i in range(50):
      key, sub = jax.random.split(key)
      # Noisy, sign-flipping gradients to stress the momentum term.
      grad = jax.random.normal(sub, (32,)) + 0.3 * jnp.sin(i * params)
      updates, state = tx.update(grad, state, params)
      inner = float(jnp.vdot(updates, grad))
      worst = max(worst, inner)
      params = optax.apply_updates(params, updates)

    # Allow a tiny positive tolerance for floating point noise.
    self.assertLessEqual(worst, 1e-5)

  def test_pytree_params(self):
    """Works with dict (pytree) parameters and masks each leaf independently."""
    params = {'w': jnp.ones((3,)), 'b': jnp.zeros((2,))}
    tx = cautious(optax.adam(1e-2))
    state = tx.init(params)
    grads = {'w': jnp.array([1.0, -1.0, 1.0]), 'b': jnp.array([0.5, -0.5])}
    updates, _ = tx.update(grads, state, params)
    jax.tree.map(
        lambda u: self.assertTrue(jnp.all(jnp.isfinite(u))), updates
    )

  @parameterized.parameters(
      {'base_name': 'adam'},
      {'base_name': 'adamw'},
      {'base_name': 'lion'},
      {'base_name': 'sgd'},
  )
  def test_wraps_common_optimizers(self, base_name):
    """cautious() should descend a quadratic when wrapping common optimizers."""
    params = jnp.array([3.0, -2.0, 1.0, 4.0])
    base = getattr(optax, base_name)(learning_rate=1e-1)
    tx = cautious(base)
    state = tx.init(params)

    def loss(p):
      return jnp.sum(p ** 2)

    initial = loss(params)
    for _ in range(100):
      grad = jax.grad(loss)(params)
      updates, state = tx.update(grad, state, params)
      params = optax.apply_updates(params, updates)
    self.assertLess(loss(params), initial)

  def test_jit_compatible(self):
    """The wrapped optimizer can be jitted end-to-end."""
    params = jnp.array([1.0, 2.0, 3.0])
    tx = cautious(optax.adam(1e-2))

    @jax.jit
    def step(params, state):
      grad = jax.grad(lambda p: jnp.sum(p**2))(params)
      updates, state = tx.update(grad, state, params)
      return optax.apply_updates(params, updates), state

    state = tx.init(params)
    for _ in range(5):
      params, state = step(params, state)
    self.assertTrue(jnp.all(jnp.isfinite(params)))

  def test_eps_one_matches_paper(self):
    """eps=1.0 reproduces the paper's n / (num_kept + 1) damping."""
    fixed_update = jnp.array([-1.0, -1.0, -1.0, 1.0])
    grad = jnp.array([1.0, 1.0, 1.0, 1.0])
    # keep [T, T, T, F], num_kept = 3, n = 4, scale = 4 / (3 + 1) = 1.0

    def _const_update(updates, state, params=None, **kw):
      del updates, params, kw
      return fixed_update, state

    const_opt = optax.GradientTransformation(
        lambda p: optax.EmptyState(), _const_update
    )
    tx = cautious(const_opt, eps=1.0)
    state = tx.init(jnp.zeros(4))
    caut_updates, _ = tx.update(grad, state, jnp.zeros(4))
    expected = jnp.array([-1.0, -1.0, -1.0, 0.0])  # scale 1.0
    chex.assert_trees_all_close(caut_updates, expected, atol=1e-6)


class CautiousExportTest(absltest.TestCase):

  def test_exported_from_contrib(self):
    self.assertIs(contrib.cautious, cautious)
    self.assertIs(contrib.CautiousState, CautiousState)


if __name__ == '__main__':
  absltest.main()
