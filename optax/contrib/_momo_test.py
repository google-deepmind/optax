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
"""Tests for momo and momo_adam."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax._src import update
from optax.contrib import _momo


def _quadratic(x):
  return jnp.sum(x**2)


class MomoTest(parameterized.TestCase):
  """Checks the weight decay step against Lemma 3.1 of the MoMo paper.

  With weight decay lambda the paper gives the closed form update
  x_{k+1} = (x_k - tau_k D_k^{-1} d_k) / (1 + alpha_k lambda). At the first
  step all running averages are known in closed form, so the expected
  parameters can be computed directly from the gradient and the loss.
  """

  @parameterized.product(weight_decay=(1e-2, 1e-1))
  def test_momo_first_step_with_weight_decay(self, weight_decay):
    lr = 1.0
    params = jnp.array([1.0, 2.0, 3.0])
    value, grads = jax.value_and_grad(_quadratic)(params)
    opt = _momo.momo(learning_rate=lr, weight_decay=weight_decay)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params, value=value)
    new_params = update.apply_updates(params, updates)

    # At the first step d_1 is the gradient, bar f_1 is the loss and gamma_1
    # is the inner product between the gradient and the parameters.
    scale = 1.0 + lr * weight_decay
    gamma = jnp.vdot(grads, params)
    numerator = scale * (value - gamma) + jnp.vdot(grads, params)
    t1 = jnp.maximum(numerator, 0.0) / jnp.vdot(grads, grads)
    tau = jnp.minimum(lr, t1)
    expected = (params - tau * grads) / scale
    test_utils.assert_trees_all_close(
        new_params, expected, rtol=1e-4, atol=1e-5
    )

  @parameterized.product(weight_decay=(1e-2, 1e-1))
  def test_momo_adam_first_step_with_weight_decay(self, weight_decay):
    lr, b1, b2, eps = 1.5, 0.9, 0.999, 1e-8
    params = jnp.array([1.0, 2.0, 3.0])
    value, grads = jax.value_and_grad(_quadratic)(params)
    opt = _momo.momo_adam(
        learning_rate=lr, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params, value=value)
    new_params = update.apply_updates(params, updates)

    # At the first step the running averages are (1 - b1) times the
    # gradient, the loss and the inner product between the gradient and the
    # parameters, and the bias corrected preconditioner is eps + |gradient|.
    scale = 1.0 + lr * weight_decay
    bc1 = 1.0 - b1
    exp_avg = bc1 * grads
    barf = bc1 * value
    gamma = bc1 * jnp.vdot(grads, params)
    precond = eps + jnp.abs(grads)
    direction = exp_avg / precond
    numerator = scale * (barf - gamma) + jnp.vdot(exp_avg, params)
    t1 = jnp.maximum(numerator, 0.0) / jnp.vdot(exp_avg, direction)
    tau = jnp.minimum(lr / bc1, t1)
    expected = (params - tau * direction) / scale
    test_utils.assert_trees_all_close(
        new_params, expected, rtol=1e-4, atol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
