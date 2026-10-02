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
"""Specific tests for `_acprop.py`, see `_common_test.py` for usual tests."""

from absl.testing import absltest
import jax.numpy as jnp
from optax._src import test_utils
from optax.contrib import _acprop


class AcpropTest(absltest.TestCase):

  def test_second_moment_uses_new_first_moment(self):
    # The documented ACProp second moment is
    #   s_t = b2 * s_{t-1} + (1 - b2) * (g_t - m_t) ** 2 + eps_root,
    # where m_t is the first moment *after* the step-t update.
    b1, b2, eps_root = 0.9, 0.999, 0.0
    grads = [
        jnp.array([1.0, -2.0]),
        jnp.array([0.5, -1.5]),
        jnp.array([-1.0, 0.3]),
    ]

    tx = _acprop.scale_by_acprop(b1=b1, b2=b2, eps=1e-16, eps_root=eps_root)
    state = tx.init(grads[0])

    m = jnp.zeros(2)
    s = jnp.zeros(2)
    for g in grads:
      m = b1 * m + (1 - b1) * g
      s = b2 * s + (1 - b2) * (g - m) ** 2 + eps_root
      _, state = tx.update(g, state, None)
      test_utils.assert_trees_all_close(state.mu, m, atol=1e-6, rtol=1e-6)
      test_utils.assert_trees_all_close(state.nu, s, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
