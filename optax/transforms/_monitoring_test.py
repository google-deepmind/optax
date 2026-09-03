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

import jax
import jax.numpy as jnp
import numpy as np
from optax import tree
from optax._src import alias
from optax._src import update
from optax.transforms import _clipping
from optax.transforms import _combining
from optax.transforms import _monitoring


class MonitoringTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
