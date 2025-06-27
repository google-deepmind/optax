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


class HooksTest(absltest.TestCase):
  def test_snapshot(self):
    """Tests that snapshot stores the correct values."""
    def f(x):
      return jnp.sum(x ** 2)

    opt_before_clip = _combining.chain(
        alias.sgd(learning_rate=0.1, momentum=0.9),
        _monitoring.snapshot('norm_before_clip', tree.norm)
    )
    opt = _combining.chain(opt_before_clip, _clipping.clip_by_global_norm(0.05))

    params = jnp.array([1., 2., 3.])
    state_aux = opt_before_clip.init(params)
    state = opt.init(params)

    # Testing for two steps to observe behavior not only after initialization
    # but also after the first update.
    for step in range(2):
      grads = jax.grad(f)(params)
      updates_before_clip, state_aux = opt_before_clip.update(grads, state_aux)
      updates, state = opt.update(grads, state)
      params = update.apply_updates(params, updates)
      with self.subTest(f'norm equals at {step=}'):
        got = tree.get(state, 'norm_before_clip')
        expected = tree.norm(updates_before_clip)
        np.testing.assert_allclose(got, expected)


if __name__ == '__main__':
  absltest.main()
