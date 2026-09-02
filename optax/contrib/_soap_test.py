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
"""Tests for optax.contrib._soap."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import numerics
from optax._src import test_utils
from optax._src import transform
from optax._src import update
from optax.contrib import _soap
import optax.tree


def _parabola_2d(dtype=jnp.float32):
  initial = jnp.zeros((4, 4), dtype=dtype)
  target = jnp.array(
      [[1.0, 2.0, -1.0, 0.5],
       [-1.0, 0.0, 2.0, 1.0],
       [0.5, -0.5, 1.0, -1.0],
       [0.0, 1.0, -0.5, 2.0]], dtype=dtype,
  )
  obj_fn = lambda p: jnp.sum(numerics.abs_sq(p - target))
  return initial, target, obj_fn


def _mixed_params(dtype=jnp.float32):
  """Dict with one 2D param (SOAP path) and one 1D param (Adam fallback)."""
  initial = {
      'w': jnp.zeros((3, 3), dtype=dtype),
      'b': jnp.zeros((3,), dtype=dtype),
  }
  target = {
      'w': jnp.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0], [-0.5, 1.0, 0.0]],
                     dtype=dtype),
      'b': jnp.array([1.0, -1.0, 0.5], dtype=dtype),
  }

  def obj_fn(params):
    return jnp.sum(numerics.abs_sq(params['w'] - target['w'])) + jnp.sum(
        numerics.abs_sq(params['b'] - target['b'])
    )

  return initial, target, obj_fn


class ScaleBySOAPTest(parameterized.TestCase):

  def test_state_shapes_2d_param(self):
    """Kronecker factors and bases have the expected shapes for 2D params."""
    m, n = 5, 3
    params = jnp.zeros((m, n))
    tx = _soap.scale_by_soap()
    state = tx.init(params)

    leaves = jax.tree.leaves(state.left_factor)
    self.assertEqual(leaves[0].shape, (m, m))
    leaves = jax.tree.leaves(state.right_factor)
    self.assertEqual(leaves[0].shape, (n, n))
    leaves = jax.tree.leaves(state.left_basis)
    self.assertEqual(leaves[0].shape, (m, m))
    leaves = jax.tree.leaves(state.right_basis)
    self.assertEqual(leaves[0].shape, (n, n))
    leaves = jax.tree.leaves(state.mu)
    self.assertEqual(leaves[0].shape, (m, n))
    leaves = jax.tree.leaves(state.nu)
    self.assertEqual(leaves[0].shape, (m, n))

  def test_state_shapes_1d_param(self):
    """Non-2D params use empty placeholder arrays for the factor fields."""
    params = jnp.zeros((7,))
    tx = _soap.scale_by_soap()
    state = tx.init(params)

    for field_name in ('left_factor', 'right_factor', 'left_basis',
                       'right_basis'):
      leaves = jax.tree.leaves(getattr(state, field_name))
      self.assertEqual(leaves[0].shape, (0,))

    leaves = jax.tree.leaves(state.mu)
    self.assertEqual(leaves[0].shape, (7,))

  def test_state_shapes_mixed_params(self):
    m, n, d = 4, 3, 5
    params = {'w': jnp.zeros((m, n)), 'b': jnp.zeros((d,))}
    tx = _soap.scale_by_soap()
    state = tx.init(params)

    lf_leaves = jax.tree.leaves(state.left_factor)
    rf_leaves = jax.tree.leaves(state.right_factor)
    # jax.tree.leaves traversal order is sorted by key
    self.assertEqual(lf_leaves[0].shape, (0,))   # 'b' is 1D
    self.assertEqual(lf_leaves[1].shape, (m, m))  # 'w' is 2D
    self.assertEqual(rf_leaves[0].shape, (0,))
    self.assertEqual(rf_leaves[1].shape, (n, n))

  def test_bases_are_orthogonal_after_update(self):
    """Eigenbases must satisfy Q Q^T ≈ I after a gradient step."""
    params = jnp.ones((4, 4))
    tx = _soap.scale_by_soap(precondition_frequency=1)
    state = tx.init(params)

    grads = jax.random.normal(jax.random.PRNGKey(0), params.shape)
    _, new_state = tx.update(grads, state)

    q_l = jax.tree.leaves(new_state.left_basis)[0]
    q_r = jax.tree.leaves(new_state.right_basis)[0]
    n = q_l.shape[0]

    self.assertEqual(q_l.shape, (n, n))
    self.assertEqual(q_r.shape, (params.shape[1], params.shape[1]))

    eye_l = q_l.T @ q_l
    eye_r = q_r.T @ q_r
    self.assertTrue(jnp.allclose(eye_l, jnp.eye(n), atol=1e-5))
    self.assertTrue(jnp.allclose(eye_r, jnp.eye(q_r.shape[0]), atol=1e-5))

  def test_kronecker_factors_are_symmetric(self):
    """Left and right Kronecker factors must remain symmetric."""
    params = jnp.zeros((5, 3))
    tx = _soap.scale_by_soap()
    state = tx.init(params)

    key = jax.random.PRNGKey(42)
    for _ in range(5):
      key, subkey = jax.random.split(key)
      grads = jax.random.normal(subkey, params.shape)
      _, state = tx.update(grads, state)

    l = jax.tree.leaves(state.left_factor)[0]
    r = jax.tree.leaves(state.right_factor)[0]
    self.assertTrue(jnp.allclose(l, l.T, atol=1e-6))
    self.assertTrue(jnp.allclose(r, r.T, atol=1e-6))

  def test_precondition_frequency_respected(self):
    """Eigenbases should only change at multiples of precondition_frequency.

    With precondition_frequency=k and count starting at 0:
      - update call i sees count=i, triggers eigh when i % k == 0.
      - So eigh runs at calls 0, k, 2k, ... (0-indexed).
      - bases_at_step[i] holds the basis after call i.
      - bases_at_step[0..k-1] are all identical (refresh only at call 0).
      - bases_at_step[k] is fresh (refresh at call k).
    """
    k = 5
    params = jnp.ones((3, 3))
    tx = _soap.scale_by_soap(precondition_frequency=k)
    state = tx.init(params)

    key = jax.random.PRNGKey(7)
    bases_at_step = []
    for _ in range(k + 2):
      key, subkey = jax.random.split(key)
      grads = jax.random.normal(subkey, params.shape)
      _, state = tx.update(grads, state)
      q_l = jax.tree.leaves(state.left_basis)[0].copy()
      bases_at_step.append(q_l)

    # Steps 1..k-1 must share the same basis as step 0 (no refresh in between).
    for i in range(1, k):
      self.assertTrue(
          jnp.allclose(bases_at_step[i], bases_at_step[0]),
          msg=f'basis unexpectedly changed at step {i}',
      )

    # Step k triggers a refresh; the new basis should differ from the previous
    # one (which was computed from a single gradient, making it very unlikely
    # to match the new one computed from k accumulated steps).
    self.assertFalse(
        jnp.allclose(bases_at_step[k], bases_at_step[k - 1]),
        msg='basis did not change at the expected precondition step',
    )

  def test_convergence_on_2d_quadratic(self):
    """SOAP converges to the minimum of a simple quadratic over 2D params."""
    initial, target, obj_fn = _parabola_2d()

    solver = _soap.soap(learning_rate=1e-2, precondition_frequency=5)
    params = initial
    state = solver.init(params)

    @jax.jit
    def step(params, state):
      grads = jax.grad(obj_fn)(params)
      updates, state = solver.update(grads, state, params)
      return update.apply_updates(params, updates), state

    for _ in range(2000):
      params, state = step(params, state)

    self.assertTrue(
        jnp.allclose(params, target, atol=0.05),
        msg=f'Max deviation: {jnp.max(jnp.abs(params - target)):.4f}',
    )

  def test_convergence_on_mixed_params(self):
    """SOAP handles dicts with both 2D and 1D params."""
    initial, target, obj_fn = _mixed_params()

    solver = _soap.soap(learning_rate=1e-2, precondition_frequency=5)
    params = initial
    state = solver.init(params)

    @jax.jit
    def step(params, state):
      grads = jax.grad(obj_fn)(params)
      updates, state = solver.update(grads, state, params)
      return update.apply_updates(params, updates), state

    for _ in range(3000):
      params, state = step(params, state)

    for key, val in target.items():
      self.assertTrue(
          jnp.allclose(params[key], val, atol=0.05),
          msg=f"key={key}, max deviation:"
              f" {jnp.max(jnp.abs(params[key] - val)):.4f}",
      )

  def test_jit_no_recompilation(self):
    """optimizer.update should not retrace on the second call."""
    params = {'w': jnp.ones((3, 3)), 'b': jnp.ones(3)}
    solver = _soap.soap(learning_rate=1e-3)
    state = solver.init(params)
    grads = jax.tree.map(jnp.ones_like, params)

    step = jax.jit(lambda p, s: solver.update(grads, s, p))
    _, state = step(params, state)

    with test_utils.log_compilations() as logs:
      _ = step(params, state)

    self.assertEmpty(logs, 'soap.update recompiled on the second call')

  @parameterized.parameters(
      {'b1': 0.9, 'b2': 0.999},
      {'b1': 0.95, 'b2': 0.99},
  )
  def test_nond_fallback_matches_adam(self, b1, b2):
    """For 1D params, scale_by_soap should produce the same updates as Adam."""
    params = jnp.array([-1.0, 2.0, 0.5])
    grads = jnp.array([0.1, -0.2, 0.3])
    eps = 1e-8

    soap_tx = _soap.scale_by_soap(b1=b1, b2=b2, eps=eps)
    # scale_by_soap normalizes b1/b2/eps to float32 internally; pass float32
    # here so both use identical float32 arithmetic for (1-b2) in the EMA.
    adam_tx = transform.scale_by_adam(
        b1=jnp.float32(b1), b2=jnp.float32(b2), eps=jnp.float32(eps)
    )

    soap_state = soap_tx.init(params)
    adam_state = adam_tx.init(params)

    soap_updates, _ = soap_tx.update(grads, soap_state)
    adam_updates, _ = adam_tx.update(grads, adam_state)

    self.assertTrue(
        jnp.allclose(soap_updates, adam_updates, atol=1e-6),
        msg=f'soap={soap_updates}, adam={adam_updates}',
    )

  def test_invalid_precondition_frequency_raises(self):
    with self.assertRaisesRegex(ValueError, 'precondition_frequency'):
      _soap.scale_by_soap(precondition_frequency=0)
    with self.assertRaisesRegex(ValueError, 'precondition_frequency'):
      _soap.scale_by_soap(precondition_frequency=-3)

  def test_weight_decay_applied(self):
    """Verify that weight_decay causes params to shrink toward zero."""
    initial = jnp.ones((3, 3)) * 5.0
    solver = _soap.soap(learning_rate=1e-2, weight_decay=0.1)
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
    tx = _soap.scale_by_soap(mu_dtype=jnp.bfloat16)
    state = tx.init(params)

    mu_leaves = jax.tree.leaves(state.mu)
    for leaf in mu_leaves:
      self.assertEqual(leaf.dtype, jnp.bfloat16)


if __name__ == '__main__':
  absltest.main()
