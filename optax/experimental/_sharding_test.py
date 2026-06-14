# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import update as optax_update
from optax.experimental import _sharding

P = jax.sharding.PartitionSpec

# Best-effort: set 8 CPU devices before JAX backend initialisation.
try:
  chex.set_n_cpu_devices(8)
except RuntimeError:
  pass

_REQUIRED_DEVICES = 8


def _make_explicit_mesh(shape, names):
  """Create an explicit mesh for testing."""
  axis_types = (jax.sharding.AxisType.Explicit,) * len(names)
  return jax.make_mesh(shape, names, axis_types=axis_types)


class ComputeEnhancedPspecTest(parameterized.TestCase):
  """Tests for _compute_enhanced_pspec."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < _REQUIRED_DEVICES:
      self.skipTest(f'requires {_REQUIRED_DEVICES} devices')
    # Mesh: data=2, model=4 → 8 devices total.
    self.mesh = _make_explicit_mesh((2, 4), ('data', 'model'))

  def _abstract(self, shape, pspec, mesh=None):
    mesh = mesh or self.mesh
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    return jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)

  def test_scalar_returns_empty(self):
    result = _sharding._compute_enhanced_pspec(self._abstract((), P()))
    self.assertEqual(result, P())

  def test_all_axes_already_used(self):
    """When all mesh axes are already assigned, nothing changes."""
    result = _sharding._compute_enhanced_pspec(
        self._abstract((8, 4), P('data', 'model'))
    )
    self.assertEqual(result, P('data', 'model'))

  def test_one_unused_axis_assigned(self):
    """One unused axis ('data', size 2) should be assigned to dim 0."""
    # 'model' (size 4) already on dim 1.
    # 'data' (size 2): dim 0 → 8 % 2 = 0 ✓, dim 1 → 4/(4) = 1, 1%2 ≠ 0 ✗.
    result = _sharding._compute_enhanced_pspec(
        self._abstract((8, 4), P(None, 'model'))
    )
    self.assertEqual(result, P('data', 'model'))

  def test_all_axes_unused_greedy(self):
    """Both axes unused: largest axis first, assigned to largest dim."""
    # 'model' (size 4) first → dim 0 (size 8, 8%4=0).
    # 'data' (size 2) next → dim 0 (8%(4*2)=0, size 8) vs dim 1 (4%2=0, size
    #   4). Picks dim 0 (larger).
    result = _sharding._compute_enhanced_pspec(
        self._abstract((8, 4), P(None, None))
    )
    self.assertEqual(result, P(('model', 'data'), None))

  def test_axis_does_not_fit(self):
    """Axes that don't fit any dimension are skipped."""
    # Shape (3, 5): 3 % 4 ≠ 0, 5 % 4 ≠ 0, 3 % 2 ≠ 0, 5 % 2 ≠ 0.
    result = _sharding._compute_enhanced_pspec(
        self._abstract((3, 5), P(None, None))
    )
    self.assertEqual(result, P(None, None))

  def test_partial_fit(self):
    """Only some unused axes fit."""
    # Shape (6, 3): 'model' (4) → 6%4≠0, 3%4≠0 → skip.
    #              'data' (2) → 6%2=0 → dim 0.
    result = _sharding._compute_enhanced_pspec(
        self._abstract((6, 3), P(None, None))
    )
    self.assertEqual(result, P('data', None))

  def test_1d_array(self):
    """Single-dimension array."""
    result = _sharding._compute_enhanced_pspec(self._abstract((8,), P(None)))
    # 'model' (4): 8%4=0. 'data' (2): 8%(4*2)=0.
    self.assertEqual(
        result,
        P(
            ('model', 'data'),
        ),
    )

  def test_prefers_largest_dimension(self):
    """When an axis fits in multiple dims, the largest dim is chosen."""
    # 3-axis mesh: a=2, b=2, c=2.
    mesh = _make_explicit_mesh((2, 2, 2), ('a', 'b', 'c'))
    # Shape (8, 4): pspec P('a', None).
    # 'b' (2): dim 0 (8%(2*2)=0, size 8) vs dim 1 (4%2=0, size 4) → dim 0.
    # 'c' (2): dim 0 (8%(4*2)=0, size 8) vs dim 1 (4%2=0, size 4) → dim 0.
    result = _sharding._compute_enhanced_pspec(
        self._abstract((8, 4), P('a', None), mesh)
    )
    self.assertEqual(result, P(('a', 'b', 'c'), None))

  def test_distributes_across_dims_when_needed(self):
    """When an axis can't stack on an existing dim, it goes to another."""
    # 3-axis mesh: a=2, b=2, c=2.
    mesh = _make_explicit_mesh((2, 2, 2), ('a', 'b', 'c'))
    # Shape (4, 6): pspec P('a', None).
    # 'b' (2): dim 0 (4%(2*2)=0, size 4) vs dim 1 (6%2=0, size 6) → dim 1.
    # 'c' (2): dim 0 (4%(4)=0, size 4) vs dim 1 (6%(2*2)≠0, nope).
    #          → dim 0 wins for 'c'.
    result = _sharding._compute_enhanced_pspec(
        self._abstract((4, 6), P('a', None), mesh)
    )
    self.assertEqual(result, P(('a', 'c'), 'b'))


class WithCustomShardingTest(absltest.TestCase):
  """Integration tests for with_custom_sharding."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < _REQUIRED_DEVICES:
      self.skipTest(f'requires {_REQUIRED_DEVICES} devices')
    self.mesh = _make_explicit_mesh((2, 4), ('data', 'model'))
    jax.sharding.set_mesh(self.mesh)

  def _make_params(self, shape=(8, 4), pspec=P(None, 'model')):
    """Create params with the given shape and sharding."""
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return jax.device_put(jnp.ones(shape, dtype=jnp.float32), sharding)

  def test_init_enhances_state_sharding(self):
    """State leaves matching param shape should get enhanced sharding."""
    params = self._make_params()
    tx = _sharding.with_custom_sharding(alias.adam(1e-3))
    state = tx.init(params)

    # The mu and nu arrays should have enhanced sharding (using 'data' axis).
    for leaf in jax.tree.leaves(state):
      if leaf.shape == (8, 4):
        leaf_pspec = jax.typeof(leaf).sharding.spec
        # 'data' should be incorporated into the sharding.
        self.assertIn('data', str(leaf_pspec))

  def test_update_produces_enhanced_sharding(self):
    """Output updates should carry the enhanced (zero-redundancy) sharding."""
    params = self._make_params()
    tx = _sharding.with_custom_sharding(alias.adam(1e-3))
    state = tx.init(params)

    grads = self._make_params()  # Same shape/sharding as params.
    updates, _ = tx.update(grads, state, params)

    # Updates should be in the enhanced sharding domain.
    for upd_leaf in jax.tree.leaves(updates):
      if upd_leaf.shape == (8, 4):
        self.assertIn('data', str(jax.typeof(upd_leaf).sharding.spec))

  def test_numerical_correctness(self):
    """Wrapped transform should produce same values as unwrapped."""
    params = self._make_params()
    grads = self._make_params()

    # Unwrapped.
    tx_plain = alias.adam(1e-3)
    state_plain = tx_plain.init(params)
    updates_plain, _ = tx_plain.update(grads, state_plain, params)

    # Wrapped.
    tx_wrapped = _sharding.with_custom_sharding(alias.adam(1e-3))
    state_wrapped = tx_wrapped.init(params)
    updates_wrapped, _ = tx_wrapped.update(grads, state_wrapped, params)

    chex.assert_trees_all_close(updates_plain, updates_wrapped, atol=1e-7)

  def test_multiple_update_steps(self):
    """Run several update steps to verify state evolution."""
    params = self._make_params()
    tx = _sharding.with_custom_sharding(alias.adam(1e-3))
    state = tx.init(params)

    rng = np.random.RandomState(42)
    for _ in range(5):
      grads_np = rng.randn(*params.shape).astype(np.float32)
      grads = jax.device_put(
          jnp.array(grads_np),
          jax.sharding.NamedSharding(self.mesh, P(None, 'model')),
      )
      updates, state = tx.update(grads, state, params)
      params = optax_update.apply_updates(params, updates)

    # Verify params are finite.
    for leaf in jax.tree.leaves(params):
      self.assertTrue(jnp.all(jnp.isfinite(leaf)))

  def test_sgd_with_momentum(self):
    """Test wrapping SGD with momentum (simpler state than Adam)."""
    params = self._make_params()
    tx = _sharding.with_custom_sharding(alias.sgd(1e-2, momentum=0.9))
    state = tx.init(params)

    grads = self._make_params()
    _, new_state = tx.update(grads, state, params)

    # SGD momentum state should have enhanced sharding.
    for leaf in jax.tree.leaves(new_state):
      if leaf.shape == (8, 4):
        self.assertIn('data', str(jax.typeof(leaf).sharding.spec))

  def test_no_unused_axes_is_identity(self):
    """If params already use all axes, wrapping should be a no-op."""
    # Params sharded across both axes: P('data', 'model').
    params = self._make_params(pspec=P('data', 'model'))

    tx_plain = alias.sgd(1e-2, momentum=0.9)
    tx_wrapped = _sharding.with_custom_sharding(alias.sgd(1e-2, momentum=0.9))

    state_plain = tx_plain.init(params)
    state_wrapped = tx_wrapped.init(params)

    # State shardings should match since no enhancement is possible.
    for plain_leaf, wrapped_leaf in zip(
        jax.tree.leaves(state_plain),
        jax.tree.leaves(state_wrapped),
    ):
      self.assertEqual(
          jax.typeof(plain_leaf).sharding.spec,
          jax.typeof(wrapped_leaf).sharding.spec,
      )


if __name__ == '__main__':
  absltest.main()
