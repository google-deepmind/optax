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
"""Tests for optax.contrib._mars."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import numerics
from optax._src import transform
from optax._src import update
from optax.contrib import _mars
import optax.tree


class ScaleByMARSTest(parameterized.TestCase):

  def test_state_shapes(self):
    """last_grad, mu, and nu should have the same shape as params."""
    params = {'w': jnp.zeros((3, 4)), 'b': jnp.zeros((4,))}
    tx = _mars.scale_by_mars()
    state = tx.init(params)

    # jax.tree.leaves traverses dicts in sorted key order, so check via the
    # state fields themselves rather than by assumed list index.
    param_shapes = [leaf.shape for leaf in jax.tree.leaves(params)]
    for field in ('last_grad', 'mu', 'nu'):
      field_shapes = [leaf.shape for leaf in jax.tree.leaves(getattr(state, field))]
      self.assertEqual(field_shapes, param_shapes, msg=f'{field} shapes mismatch')

  def test_last_grad_stored_after_update(self):
    """After an update step the state should hold the gradient from that step."""
    params = jnp.array([1.0, -2.0, 0.5])
    grads = jnp.array([0.3, -0.1, 0.7])
    tx = _mars.scale_by_mars()
    state = tx.init(params)
    _, new_state = tx.update(grads, state)
    # last_grad should now equal the gradient we just applied.
    self.assertTrue(
        jnp.allclose(
            jax.tree.leaves(new_state.last_grad)[0],
            grads.astype(jnp.float32),
        ),
        msg='last_grad not updated correctly',
    )

  def test_gamma_zero_matches_adam(self):
    """With gamma=0 the variance-reduction term vanishes and MARS == Adam."""
    params = jnp.array([-1.0, 2.0, 0.5])
    grads = jnp.array([0.1, -0.2, 0.3])
    b1, b2, eps = 0.9, 0.999, 1e-8

    mars_tx = _mars.scale_by_mars(
        b1=b1, b2=b2, eps=eps, gamma=0.0, clip_threshold=None
    )
    # scale_by_mars normalizes to float32 internally; pass float32 to Adam.
    adam_tx = transform.scale_by_adam(
        b1=jnp.float32(b1), b2=jnp.float32(b2), eps=jnp.float32(eps)
    )

    mars_state = mars_tx.init(params)
    adam_state = adam_tx.init(params)

    mars_updates, _ = mars_tx.update(grads, mars_state)
    adam_updates, _ = adam_tx.update(grads, adam_state)

    self.assertTrue(
        jnp.allclose(mars_updates, adam_updates, atol=1e-6),
        msg=f'mars={mars_updates}, adam={adam_updates}',
    )

  def test_clipping_reduces_large_c(self):
    """When the corrected gradient norm exceeds clip_threshold it is rescaled."""
    # Use a large gamma so the correction term inflates c well past 1.0.
    params = jnp.ones((4,))
    grads = jnp.ones((4,)) * 10.0  # large gradient, last_grad initialised to 0

    tx = _mars.scale_by_mars(gamma=5.0, clip_threshold=1.0)
    state = tx.init(params)
    _, new_state = tx.update(grads, state)

    # Retrieve the mu after the first step — it was updated from a clipped c,
    # so its L2 norm should be no larger than 1.0 (before bias correction).
    mu_leaf = jax.tree.leaves(new_state.mu)[0]
    # mu = (1-b1) * c_clipped; since c_clipped is unit norm, |mu| ≈ (1-b1)
    b1 = 0.95
    self.assertLessEqual(
        float(jnp.sum(jnp.square(mu_leaf))),
        (1.0 - b1) ** 2 + 1e-5,
        msg='clipping did not reduce the update magnitude',
    )

  def test_no_clipping_when_disabled(self):
    """With clip_threshold=None, large c_t values are not clamped."""
    params = jnp.ones((2,))
    grads = jnp.ones((2,)) * 100.0  # very large gradient

    tx_clipped = _mars.scale_by_mars(gamma=1.0, clip_threshold=1.0)
    tx_unclipped = _mars.scale_by_mars(gamma=1.0, clip_threshold=None)
    state_c = tx_clipped.init(params)
    state_u = tx_unclipped.init(params)

    u_clipped, _ = tx_clipped.update(grads, state_c)
    u_unclipped, _ = tx_unclipped.update(grads, state_u)

    # Without clipping the update magnitude should be strictly larger.
    self.assertGreater(
        float(jnp.sum(jnp.abs(u_unclipped))),
        float(jnp.sum(jnp.abs(u_clipped))),
        msg='disabling clipping should increase update magnitude for large grads',
    )

  def test_variance_reduction_uses_last_grad(self):
    """The second-step update should differ from a plain Adam step because
    MARS subtracts the previous gradient in the c_t calculation."""
    params = jnp.zeros((3,))
    g1 = jnp.array([1.0, 0.0, 0.0])
    g2 = jnp.array([0.0, 1.0, 0.0])

    # MARS step 2 with gamma > 0
    mars_tx = _mars.scale_by_mars(gamma=0.5, clip_threshold=None)
    mars_state = mars_tx.init(params)
    _, mars_state = mars_tx.update(g1, mars_state)
    mars_upd, _ = mars_tx.update(g2, mars_state)

    # Adam step 2 (gamma=0 branch of MARS)
    adam_tx = _mars.scale_by_mars(gamma=0.0, clip_threshold=None)
    adam_state = adam_tx.init(params)
    _, adam_state = adam_tx.update(g1, adam_state)
    adam_upd, _ = adam_tx.update(g2, adam_state)

    # With gamma > 0 the direction of the update should differ because c_t
    # for MARS includes -last_grad = -g1, which has a nonzero first component.
    self.assertFalse(
        jnp.allclose(mars_upd, adam_upd),
        msg='gamma > 0 should cause MARS to differ from Adam on the second step',
    )

  def test_convergence_on_quadratic(self):
    """MARS should converge to the minimum of a simple quadratic."""
    initial = jnp.zeros((4, 4), dtype=jnp.float32)
    target = jnp.array(
        [[1.0, -1.0, 0.5, 0.0],
         [0.0, 2.0, -1.0, 0.5],
         [-0.5, 1.0, 0.0, -1.0],
         [1.0, 0.0, -0.5, 2.0]], dtype=jnp.float32,
    )
    obj_fn = lambda p: jnp.sum(numerics.abs_sq(p - target))

    solver = _mars.mars(learning_rate=3e-3, gamma=0.025)
    params = initial
    state = solver.init(params)

    @jax.jit
    def step(params, state):
      grads = jax.grad(obj_fn)(params)
      updates, state = solver.update(grads, state, params)
      return update.apply_updates(params, updates), state

    for _ in range(3000):
      params, state = step(params, state)

    self.assertTrue(
        jnp.allclose(params, target, atol=0.05),
        msg=f'Max deviation: {jnp.max(jnp.abs(params - target)):.4f}',
    )

  def test_convergence_on_mixed_params(self):
    """MARS handles dicts with params of different ranks."""
    initial = {'w': jnp.zeros((3, 3)), 'b': jnp.zeros((3,))}
    target = {
        'w': jnp.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0], [-0.5, 1.0, 0.0]]),
        'b': jnp.array([1.0, -1.0, 0.5]),
    }
    obj_fn = lambda p: (
        jnp.sum(numerics.abs_sq(p['w'] - target['w']))
        + jnp.sum(numerics.abs_sq(p['b'] - target['b']))
    )

    solver = _mars.mars(learning_rate=3e-3, gamma=0.025)
    params = initial
    state = solver.init(params)

    @jax.jit
    def step(params, state):
      grads = jax.grad(obj_fn)(params)
      updates, state = solver.update(grads, state, params)
      return update.apply_updates(params, updates), state

    for _ in range(3000):
      params, state = step(params, state)

    for key in target:
      self.assertTrue(
          jnp.allclose(params[key], target[key], atol=0.05),
          msg=f"key={key}, max dev: {jnp.max(jnp.abs(params[key] - target[key])):.4f}",
      )

  def test_jit_no_recompilation(self):
    """mars.update should not retrace on the second call."""
    from optax._src import test_utils  # pylint: disable=g-import-not-at-top

    params = {'w': jnp.ones((3, 3)), 'b': jnp.ones(3)}
    solver = _mars.mars(learning_rate=3e-3)
    state = solver.init(params)
    grads = jax.tree.map(jnp.ones_like, params)

    step = jax.jit(lambda p, s: solver.update(grads, s, p))
    _, state = step(params, state)

    with test_utils.log_compilations() as logs:
      _ = step(params, state)

    self.assertEmpty(logs, 'mars.update recompiled on the second call')

  def test_weight_decay_applied(self):
    """Weight decay should cause parameter magnitudes to decrease."""
    initial = jnp.ones((3, 3)) * 5.0
    solver = _mars.mars(learning_rate=3e-3, weight_decay=0.1)
    params = initial
    state = solver.init(params)
    zero_grads = jnp.zeros_like(params)

    @jax.jit
    def step(p, s):
      u, s = solver.update(zero_grads, s, p)
      return update.apply_updates(p, u), s

    for _ in range(100):
      params, state = step(params, state)

    self.assertTrue(
        jnp.all(jnp.abs(params) < jnp.abs(initial)),
        msg='weight decay did not reduce parameter magnitude',
    )

  def test_mu_dtype_reduces_mu_precision(self):
    """mu should be stored in mu_dtype when specified."""
    params = jnp.ones((3, 3))
    tx = _mars.scale_by_mars(mu_dtype=jnp.bfloat16)
    state = tx.init(params)

    # Run one step so mu is non-zero.
    grads = jnp.ones_like(params)
    _, state = tx.update(grads, state)

    mu_leaves = jax.tree.leaves(state.mu)
    for leaf in mu_leaves:
      self.assertEqual(leaf.dtype, jnp.bfloat16)

  @parameterized.parameters(
      {'gamma': 0.01, 'b1': 0.9, 'b2': 0.999},
      {'gamma': 0.05, 'b1': 0.95, 'b2': 0.99},
  )
  def test_state_count_increments(self, gamma, b1, b2):
    """count should advance by one each update step."""
    params = jnp.zeros((2, 2))
    tx = _mars.scale_by_mars(b1=b1, b2=b2, gamma=gamma)
    state = tx.init(params)
    self.assertEqual(int(state.count), 0)

    grads = jnp.ones_like(params)
    _, state = tx.update(grads, state)
    self.assertEqual(int(state.count), 1)

    _, state = tx.update(grads, state)
    self.assertEqual(int(state.count), 2)


if __name__ == '__main__':
  absltest.main()
