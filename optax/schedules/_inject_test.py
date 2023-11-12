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
"""Tests for `inject.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import clipping
from optax._src import transform
from optax._src import wrappers
from optax.schedules import _inject
from optax.schedules import _schedule
from optax.tree_utils import _state_utils


class InjectHyperparamsTest(chex.TestCase):
  """Tests for the inject_hyperparams wrapper."""

  @chex.all_variants
  def test_updates(self):
    optim = _inject.inject_hyperparams(transform.scale)(  # stateless
        step_size=_schedule.piecewise_constant_schedule(
            3.0, {1: 5, 7: 2, 12: 1.5}))

    params = [jnp.zeros([], dtype=jnp.float32)]
    state = self.variant(optim.init)(params)

    # A no-op change, to verify that tree map works.
    state = _state_utils.tree_map_params(optim, lambda v: v, state)

    update_fn = self.variant(optim.update)
    expected_step_size = [3.0]*2 + [15.0]*6 + [30.0]*5 + [45.0]*3

    grads = [jnp.ones([], dtype=jnp.float32)]
    for i in range(15):
      updates, state = update_fn(grads, state, params=params)
      np.testing.assert_almost_equal(updates[0], expected_step_size[i+1])

  @chex.all_variants
  def test_hyperparams_state(self):
    optim = _inject.inject_hyperparams(transform.trace)(  # stateful
        decay=_schedule.piecewise_constant_schedule(
            0.8, {3: 0.5, 9: 1.25}),
        nesterov=True)

    params = [jnp.zeros([2, 3]) for _ in range(3)]
    state = self.variant(optim.init)(params)
    update_fn = self.variant(optim.update)

    expected_mom = [0.8]*4 + [0.4]*6 + [0.5]*2
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for i in range(12):
      np.testing.assert_almost_equal(state.hyperparams['decay'],
                                     expected_mom[i])
      _, state = update_fn(grads, state)

    np.testing.assert_almost_equal(state.hyperparams['decay'],
                                   expected_mom[-1])

  @chex.all_variants
  def test_constant_hyperparams(self):
    optim = _inject.inject_hyperparams(transform.scale_by_adam)(b1=0., b2=0.)

    params = [jnp.zeros([2, 3]) for _ in range(3)]
    state = self.variant(optim.init)(params)
    update_fn = self.variant(optim.update)

    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(5):
      updates, state = update_fn(grads, state, params)
      np.testing.assert_almost_equal(state.hyperparams['b1'], 0.0)
      np.testing.assert_almost_equal(state.hyperparams['b2'], 0.0)
      np.testing.assert_almost_equal(state.hyperparams['eps'], 1e-8)
      np.testing.assert_almost_equal(state.hyperparams['eps_root'], 0.0)
      assert 'eps' in state.hyperparams
      chex.assert_trees_all_close(updates, grads)

  @chex.all_variants
  def test_overriding_hyperparam(self):
    optim = _inject.inject_hyperparams(clipping.clip_by_global_norm)(0.1)
    params = jnp.zeros((3, 5, 7))
    state = self.variant(optim.init)(params)
    update_fn = self.variant(optim.update)

    grads = jnp.ones_like(params)
    for i in range(5):
      state.hyperparams['max_norm'] = i
      updates, state = update_fn(grads, state)
      assert np.isclose(jnp.linalg.norm(updates.ravel()), i)

  @chex.all_variants
  @parameterized.named_parameters(('string', 'mask'), ('list', ['mask']))
  def test_static_args(self, static_args):
    @functools.partial(_inject.inject_hyperparams, static_args=static_args)
    def custom_optim(learning_rate, mask):
      return wrappers.masked(transform.scale(-learning_rate), mask)

    optim = custom_optim(
        0.1, functools.partial(jax.tree_util.tree_map, lambda x: x.ndim > 1))
    params = [jnp.ones((1, 2)), jnp.ones(2), jnp.ones((1, 1, 1))]
    grads = params
    state = self.variant(optim.init)(params)
    updates, state = self.variant(optim.update)(grads, state)
    expected_updates = jax.tree_util.tree_map(
        lambda x: -0.1 * x if x.ndim > 1 else x, grads)

    assert set(state.hyperparams.keys()) == {'learning_rate'}, state.hyperparams
    chex.assert_trees_all_close(updates, expected_updates)

  @chex.all_variants
  @parameterized.named_parameters(('one_arg', 'b1'), ('two_arg', ['b1', 'b2']))
  def test_numeric_static_args(self, static_args):
    optim = _inject.inject_hyperparams(
        transform.scale_by_adam, static_args=static_args)(b1=0.9, b2=0.95)

    params = [jnp.ones((1, 2)), jnp.ones(2), jnp.ones((1, 1, 1))]
    grads = params
    state = self.variant(optim.init)(params)
    _, state = self.variant(optim.update)(grads, state)

    assert not set(state.hyperparams.keys()).intersection(set(static_args))

  @chex.all_variants
  @parameterized.named_parameters(
      ('bf16hyp f32param bf16grad', jnp.bfloat16, jnp.float32, jnp.bfloat16),
      ('bf16hyp f32param f32_grads', jnp.bfloat16, jnp.float32, jnp.float32),
      ('f32hyp bf16param bf16grad', jnp.float32, jnp.bfloat16, jnp.bfloat16),
      ('f32hyp f32param bf16grad', jnp.float32, jnp.float32, jnp.bfloat16),
      ('f32hyp bf16param f32grad', jnp.float32, jnp.bfloat16, jnp.float32),
      )
  def test_hyperparam_dtypes(self,
                             hyperparam_dtype,
                             param_dtype,
                             grad_dtype):
    """Tests that hyperparam dtype override works as desired."""
    optim = _inject.inject_hyperparams(
        transform.scale_by_adam,
        hyperparam_dtype=hyperparam_dtype)(b1=0.9, b2=0.95)

    params = [jnp.ones((1, 2), dtype=param_dtype),
              jnp.ones(2, dtype=param_dtype),
              jnp.ones((1, 1, 1), dtype=param_dtype)]
    grads = jax.tree_map(lambda x: x.astype(grad_dtype), params)
    state = self.variant(optim.init)(params)
    # Check that the hyperparams are overridden
    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

    _, state = self.variant(optim.update)(grads, state)

    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

  @parameterized.named_parameters(('string', 'lr'), ('list', ['lr']))
  def test_static_args_error(self, static_args):
    with self.assertRaises(ValueError):
      _inject.inject_hyperparams(transform.scale, static_args=static_args)

  @chex.all_variants
  def test_inject_hyperparams_starts_with_step_count_zero(self):
    """Checks that inject_hyperparams uses step count 0 in the first update."""
    # See also: https://github.com/deepmind/optax/issues/415.
    opt = _inject.inject_hyperparams(transform.scale)(lambda count: count)
    params = jnp.zeros(3)
    grads = jnp.array([-1, 0, 1])
    updates, _ = self.variant(opt.update)(grads, opt.init(params))
    np.testing.assert_array_equal(updates, np.zeros(3))


if __name__ == '__main__':
  absltest.main()
