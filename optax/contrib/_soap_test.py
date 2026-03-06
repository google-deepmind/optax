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
"""Tests for optax.contrib.soap."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax.contrib import _soap
from optax._src import test_utils
from optax._src import update

jax.config.update('jax_enable_x64', True)


def _setup_quadratic_target(dtype):
  # A simple target: f(X) = ||X - X_target||_F^2
  initial_params = jnp.zeros((3, 4), dtype=dtype)
  final_params = jnp.array([
      [1.0, -1.0, 2.0, 0.5],
      [-2.0, 3.0, -1.0, 1.0],
      [0.0, 1.0, 4.0, -2.0]
  ], dtype=dtype)

  def obj_fn(params):
    return jnp.sum(jnp.square(params - final_params))

  return initial_params, final_params, obj_fn


class SoapTest(parameterized.TestCase):

  def test_initialization(self):
    """Tests that the preconditioning logic initializes state correctly."""
    params = {
        '1d': jnp.ones((10,)),
        '2d': jnp.ones((4, 5)),
        '3d': jnp.ones((2, 3, 4)),
    }
    opt = _soap.scale_by_soap(max_precond_dim=4, precondition_1d=True)
    state = opt.init(params)

    # Check 1d param (exceeds max_precond_dim=4, so it gets None)
    self.assertEqual(state.gg['1d'].matrices, (None,))
    self.assertEqual(state.q['1d'].matrices, (None,))

    # Check 2d param (dim 0 is <= 4, dim 1 is > 4)
    self.assertEqual(len(state.gg['2d'].matrices), 2)
    self.assertEqual(state.gg['2d'].matrices[0].shape, (4, 4))
    self.assertIsNone(state.gg['2d'].matrices[1])

    # Check 3d param (all dims <= 4)
    self.assertEqual(len(state.gg['3d'].matrices), 3)
    self.assertEqual(state.gg['3d'].matrices[0].shape, (2, 2))
    self.assertEqual(state.gg['3d'].matrices[1].shape, (3, 3))
    self.assertEqual(state.gg['3d'].matrices[2].shape, (4, 4))

  @parameterized.product(
      dtype=[jnp.float32],
      precondition_frequency=[1, 5]
  )
  def test_optimization_convergence(self, dtype, precondition_frequency):
    """Tests that the optimizer successfully minimizes a quadratic objective."""
    initial_params, final_params, obj_fn = _setup_quadratic_target(dtype)

    learning_rate = 0.1
    opt = _soap.soap(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.999,
        precondition_frequency=precondition_frequency,
    )

    params = initial_params
    state = jax.jit(opt.init)(params)

    @jax.jit
    def step(params, state):
      loss, grads = jax.value_and_grad(obj_fn)(params)
      updates, new_state = opt.update(grads, state, params=params)
      new_params = update.apply_updates(params, updates)
      return new_params, new_state, loss

    for _ in range(200):
      params, state, _ = step(params, state)

    test_utils.assert_trees_all_close(
        params, final_params, atol=2e-1, rtol=2e-1
    )

  def test_qr_dtype_respect(self):
    """Tests that qr_dtype is respected for preconditioning tracking."""
    params = jnp.ones((3, 3), dtype=jnp.float32)
    qr_dtype = jnp.float64
    opt = _soap.scale_by_soap(qr_dtype=qr_dtype)
    state = opt.init(params)

    self.assertEqual(state.gg.matrices[0].dtype, qr_dtype)
    self.assertEqual(state.q.matrices[0].dtype, qr_dtype)


if __name__ == '__main__':
  absltest.main()
