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

"""Tests for monitoring and debugging in optax."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
from optax import tree
from optax._src import alias
from optax._src import update
from optax._src import wrappers
from optax.transforms import _clipping
from optax.transforms import _combining
from optax.transforms import _monitoring


class MonitoringTest(parameterized.TestCase):

  def test_snapshot(self):
    """Tests that snapshot stores the correct values."""

    def f(x):
      return jnp.sum(x**2)

    opt_before_clip = _combining.chain(
        alias.sgd(learning_rate=0.1, momentum=0.9),
        _monitoring.snapshot('norm_before_clip', tree.norm),
    )
    opt = _combining.chain(opt_before_clip, _clipping.clip_by_global_norm(0.05))

    params = jnp.array([1.0, 2.0, 3.0])
    state_aux = opt_before_clip.init(params)
    state = opt.init(params)

    # Testing for two steps to observe behavior not only after initialization
    # but also after the first update.
    for step in range(1, 3):
      grads = jax.grad(f)(params)
      updates_before_clip, state_aux = opt_before_clip.update(grads, state_aux)
      updates, state = opt.update(grads, state)
      params = update.apply_updates(params, updates)
      with self.subTest(f'norms equal at {step=}'):
        got = tree.get(state, 'norm_before_clip')
        expected = tree.norm(updates_before_clip)
        np.testing.assert_allclose(got, expected)

  def test_monitor(self):
    """Tests that monitor stores the correct values."""

    def f(x):
      return jnp.sum(x**2)

    ema_decay = 0.9
    debias = True
    opt_before_clip = _combining.chain(
        alias.sgd(learning_rate=0.1, momentum=0.9),
        _monitoring.monitor({
            'norm_before_clip': tree.norm,
            'norm_before_clip_ema': _monitoring.measure_with_ema(
                tree.norm, ema_decay, debias
            ),
        }),
    )
    opt = _combining.chain(opt_before_clip, _clipping.clip_by_global_norm(0.05))

    params = jnp.array([1.0, 2.0, 3.0])
    state_aux = opt_before_clip.init(params)
    state = opt.init(params)

    ema_norm = 0.0
    for step in range(1, 4):
      grads = jax.grad(f)(params)
      updates_before_clip, state_aux = opt_before_clip.update(grads, state_aux)
      updates, state = opt.update(grads, state)
      params = update.apply_updates(params, updates)
      norm_before_clip = tree.norm(updates_before_clip)
      with self.subTest(f'norms equal at {step=}'):
        np.testing.assert_allclose(
            tree.get(state, 'norm_before_clip'), norm_before_clip
        )

      ema_norm = ema_decay * ema_norm + (1 - ema_decay) * norm_before_clip
      ema_norm_debiased = ema_norm / (1 - ema_decay**step)
      with self.subTest(f'ema norms equal at {step=}'):
        np.testing.assert_allclose(
            tree.get(state, 'norm_before_clip_ema'),
            ema_norm_debiased,
            rtol=1e-5,
        )

  @parameterized.parameters(jnp.float32, jnp.bfloat16, jnp.float16)
  def test_monitor_state_structure_stability(self, dtype):
    """Tests monitor state stability, zero-init, and lax.scan equivalence."""

    def f(x):
      return jnp.sum(x**2)

    params = jnp.array([1.0, 2.0, 3.0], dtype=dtype)
    opt = _combining.chain(
        alias.sgd(0.1, momentum=0.9),
        _monitoring.monitor({
            'norm': tree.norm,
            'norm_ema': _monitoring.measure_with_ema(tree.norm, decay=0.9),
        }),
    )

    state = opt.init(params)
    grads = jax.grad(f)(params)
    _, new_state = opt.update(grads, state)

    self.assertEqual(
        jax.tree.structure(state),
        jax.tree.structure(new_state),
    )
    for init_leaf, upd_leaf in zip(
        jax.tree.leaves(state), jax.tree.leaves(new_state)
    ):
      self.assertEqual(init_leaf.shape, upd_leaf.shape)
      self.assertEqual(init_leaf.dtype, upd_leaf.dtype)
      self.assertEqual(init_leaf.weak_type, upd_leaf.weak_type)

    monitor_state = state[1]
    self.assertEqual(float(monitor_state.measurements['norm']), 0.0)
    self.assertEqual(float(monitor_state.measurements['norm_ema']), 0.0)
    ema_state = monitor_state.measure_states[1][1]
    self.assertEqual(int(ema_state.count), 0)

    def step(carry, _):
      p, s = carry
      g = jax.grad(f)(p)
      u, s = opt.update(g, s)
      return (update.apply_updates(p, u), s), (
          tree.get(s, 'norm'),
          tree.get(s, 'norm_ema'),
      )

    (scan_p, _), (scan_norms, scan_emas) = jax.lax.scan(
        step, (params, state), length=3
    )

    loop_p, loop_s = params, state
    loop_norms, loop_emas = [], []
    for _ in range(3):
      g = jax.grad(f)(loop_p)
      u, loop_s = opt.update(g, loop_s)
      loop_p = update.apply_updates(loop_p, u)
      loop_norms.append(tree.get(loop_s, 'norm'))
      loop_emas.append(tree.get(loop_s, 'norm_ema'))

    tol = 1e-2 if dtype in (jnp.bfloat16, jnp.float16) else 1e-5
    np.testing.assert_allclose(
        np.asarray(scan_p, dtype=np.float32),
        np.asarray(loop_p, dtype=np.float32),
        rtol=tol,
    )
    np.testing.assert_allclose(
        np.asarray(scan_norms, dtype=np.float32),
        np.asarray(jnp.stack(loop_norms), dtype=np.float32),
        rtol=tol,
    )
    np.testing.assert_allclose(
        np.asarray(scan_emas, dtype=np.float32),
        np.asarray(jnp.stack(loop_emas), dtype=np.float32),
        rtol=tol,
    )

  def test_monitor_wrappers_and_map_params(self):
    """Tests monitor with apply_if_finite, MultiSteps, and tree.map_params."""
    params = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
    grads = jax.tree.map(lambda x: 2.0 * x, params)
    opt = _combining.chain(
        alias.sgd(0.1, momentum=0.9),
        _monitoring.monitor({
            'norm': tree.norm,
            'norm_ema': _monitoring.measure_with_ema(tree.norm, decay=0.9),
        }),
    )

    wrapped_finite = wrappers.apply_if_finite(opt, max_consecutive_errors=2)
    wf_state = wrapped_finite.init(params)
    _, wf_new_state = wrapped_finite.update(grads, wf_state)
    self.assertEqual(
        jax.tree.structure(wf_state), jax.tree.structure(wf_new_state)
    )

    multisteps = wrappers.MultiSteps(opt, every_k_schedule=2)
    ms_state = multisteps.init(params)
    _, ms_new_state = multisteps.update(grads, ms_state, params)
    self.assertEqual(
        jax.tree.structure(ms_state), jax.tree.structure(ms_new_state)
    )

    state = opt.init(params)
    _, new_state = opt.update(grads, state)
    mapped_state = tree.map_params(opt, lambda p: p * 2.0, new_state)
    self.assertEqual(
        jax.tree.structure(new_state), jax.tree.structure(mapped_state)
    )

  def test_monitor_zero_init_params_and_type_error(self):
    """Tests zero-initialized params avoid NaNs and TypeErrors propagate."""
    zero_params = jnp.zeros((3,), dtype=jnp.float32)
    opt = _combining.chain(
        alias.sgd(0.1),
        _monitoring.monitor({'unit': lambda u: u / tree.norm(u)}),
    )
    state = opt.init(zero_params)
    unit_init = state[1].measurements['unit']
    self.assertTrue(bool(jnp.all(jnp.isfinite(unit_init))))
    np.testing.assert_array_equal(unit_init, jnp.zeros_like(zero_params))

    bad_opt = _monitoring.monitor({'bad': lambda u: u + 'not_an_array'})
    with self.assertRaises(TypeError):
      bad_opt.init(zero_params)


if __name__ == '__main__':
  absltest.main()
