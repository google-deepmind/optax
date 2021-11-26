# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `complex_valued.py`."""

from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import alias
from optax._src import complex_valued
from optax._src import update


def _loss_fun_complex_to_real(z):
  return (z.conj() * z).real.sum()


def _loss_fun_real_to_real(params):
  x, y = params
  return _loss_fun_complex_to_real(x + y * 1j)


class ComplexValuedTest(chex.TestCase):

  @chex.all_variants
  def test_split_complex(self):

    def do_update(loss_fun, optimizer, params, opt_state):
      loss, grads = jax.value_and_grad(loss_fun)(params)
      # Complex gradients need to be conjugated before being added to parameters
      grads = jax.tree_map(lambda x: x.conj(), grads)
      updates, opt_state = self.variant(optimizer.update)(grads, opt_state)
      params = update.apply_updates(params, updates)
      return loss, grads, params, opt_state

    # Random initial parameters
    x = jnp.asarray(np.random.randn(3))
    y = jnp.asarray(np.random.randn(3))
    z = x + y * 1j

    optimizer = alias.adam(learning_rate=1e-2)
    optimizer_complex = complex_valued.split_complex(optimizer)
    opt_state = self.variant(optimizer.init)((x, y))
    opt_state_complex = self.variant(optimizer_complex.init)(z)

    # Check that the loss, the gradients, and the parameters are the same for
    # real-to-real and complex-to-real loss functions in each step
    for _ in range(3):
      loss, (gx, gy), (x, y), opt_state = do_update(
          _loss_fun_real_to_real, optimizer, (x, y), opt_state)
      loss_complex, gz, z, opt_state_complex = do_update(
          _loss_fun_complex_to_real, optimizer_complex, z, opt_state_complex)
      np.testing.assert_allclose(loss, loss_complex)
      np.testing.assert_allclose(gx + gy * 1j, gz)
      np.testing.assert_allclose(x + y * 1j, z)


if __name__ == '__main__':
  absltest.main()
