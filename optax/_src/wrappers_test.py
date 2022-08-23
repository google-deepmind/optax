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
"""Tests for `wrappers.py`."""

import copy

from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import combine
from optax._src import constrain
from optax._src import transform
from optax._src import update
from optax._src import wrappers
import tree


def _build_sgd():
  return alias.sgd(1.)


def _build_stateful_sgd():
  # This SGD behaves like _build_sgd but also tests the optimizer state. The
  # momentum is set to zero rather than None so that the momentum terms are
  # calculated, but do not change the results.
  return alias.sgd(1., momentum=0.)


class WrappersTest(parameterized.TestCase):

  def test_flatten(self):
    def init_params():
      return (jnp.array([1., 2.]), jnp.array([3., 4.]))

    per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

    # First calculate new params without flattening
    optax_sgd_params = init_params()
    sgd = alias.sgd(1e-2, 0.0)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, state_sgd = sgd.update(per_step_updates, state_sgd)
    sgd_params_no_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # And now calculate new params with flattening
    optax_sgd_params = init_params()
    sgd = wrappers.flatten(sgd)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, state_sgd = sgd.update(per_step_updates, state_sgd)
    sgd_params_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # Test that both give the same result
    chex.assert_tree_all_close(
        sgd_params_no_flatten, sgd_params_flatten, atol=1e-7, rtol=1e-7)

  @chex.variants(with_jit=True, without_jit=True, with_pmap=True)
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
  )
  def test_apply_if_finite(self, opt_builder):
    one = jnp.ones([])
    nan = jnp.array(jnp.nan)
    def fn(x):
      return x * hk.get_parameter('p', [], init=hk.initializers.Constant(0.))

    fn = hk.without_apply_rng(hk.transform(fn))
    params = fn.init(jax.random.PRNGKey(1905), one)
    opt = wrappers.apply_if_finite(opt_builder(), 2)
    state = opt.init(params)
    grads_fn = jax.grad(self.variant(fn.apply))
    # Do one successful param update
    grads = grads_fn(params, one)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    # We know exactly what should be the value of params since we are
    # effectively using sgd in all cases.
    self.assertEqual(-1., float(jax.tree_util.tree_flatten(params)[0][0]))
    self.assertTrue(bool(state.last_finite))
    # Check 2 rejected param updates
    for step in range(2):
      grads = grads_fn(params, nan)
      updates, state = opt.update(grads, state, params)
      params = update.apply_updates(params, updates)
      self.assertEqual(-1., float(jax.tree_util.tree_flatten(params)[0][0]))
      self.assertFalse(bool(state.last_finite))
      self.assertEqual(step + 1, int(state.notfinite_count))
    # Next successful param update
    grads = grads_fn(params, one)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    self.assertEqual(-2., float(jax.tree_util.tree_flatten(params)[0][0]))
    self.assertTrue(bool(state.last_finite))
    # Again 2 rejected param updates
    for step in range(2):
      grads = grads_fn(params, nan)
      updates, state = opt.update(grads, state, params)
      params = update.apply_updates(params, updates)
      self.assertEqual(-2., float(jax.tree_util.tree_flatten(params)[0][0]))
      self.assertFalse(bool(state.last_finite))
      self.assertEqual(step + 1, int(state.notfinite_count))
    # Next param update with NaN is accepted since we reached maximum
    grads = grads_fn(params, nan)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    self.assertTrue(bool(jnp.isnan(jax.tree_util.tree_flatten(params)[0][0])))
    self.assertEqual(5, int(state.total_notfinite))

  def test_apply_if_finite_pmap(self):
    # Unlike in `test_apply_if_finite`:
    # * pmap is applied to the gradient computation and the optimisation;
    # * the NaNs are caused inside the function and do not come from the inputs.
    half = jnp.ones([1]) / 2.
    two = jnp.ones([1]) * 2.  # Causes a NaN in arctanh
    def fn(x):
      return jnp.arctanh(x) * hk.get_parameter(
          'p', [], init=hk.initializers.Constant(0.))
    fn = hk.without_apply_rng(hk.transform(fn))

    opt = wrappers.apply_if_finite(alias.sgd(1.), 2)
    def fn_update(params, opt_state, x):
      grads = jax.grad(fn.apply)(params, x)
      grads = jax.lax.psum(grads, axis_name='i')
      updates, new_opt_state = opt.update(grads, opt_state, params)
      new_params = update.apply_updates(params, updates)
      return new_params, new_opt_state
    fn_update = jax.pmap(fn_update, axis_name='i')

    params = fn.init(jax.random.PRNGKey(1905), half)
    opt_state = opt.init(params)
    params = jax.tree_util.tree_map(lambda x: x[None], params)
    opt_state = jax.tree_util.tree_map(lambda x: x[None], opt_state)
    # Do one successful param update
    params, opt_state = fn_update(params, opt_state, half)
    self.assertTrue(bool(opt_state.last_finite))
    # Check 2 rejected param updates
    for step in range(2):
      params, opt_state = fn_update(params, opt_state, two)
      self.assertFalse(bool(opt_state.last_finite))
      self.assertEqual(step + 1, int(opt_state.notfinite_count))
    # Next successful param update
    params, opt_state = fn_update(params, opt_state, half)
    self.assertTrue(bool(opt_state.last_finite))
    # Again 2 rejected param updates
    for step in range(2):
      params, opt_state = fn_update(params, opt_state, two)
      self.assertFalse(bool(opt_state.last_finite))
      self.assertEqual(step + 1, int(opt_state.notfinite_count))
    # Next param update with NaN is accepted since we reached maximum
    params, opt_state = fn_update(params, opt_state, two)
    self.assertEqual(5, int(opt_state.total_notfinite))

  @chex.variants(with_jit=True, without_jit=True, with_pmap=True)
  def test_multi_steps(self):
    batch_size = 32
    x_size = 7
    # Parameters should be updated only every `k_steps` optimisation steps.
    k_steps = 4
    data = jnp.ones([batch_size, x_size])

    def get_loss(x):
      loss = jnp.sum(hk.Linear(10)(x)**2)
      return loss

    loss_init, loss_apply = hk.without_apply_rng(hk.transform(get_loss))
    params = loss_init(jax.random.PRNGKey(1915), data)

    ms_opt = wrappers.MultiSteps(
        # Use a non-trivial inner optimiser:
        # * it has a state,
        # * it requires the params for the update.
        combine.chain(transform.scale_by_adam(),
                      transform.additive_weight_decay(1e-2),
                      transform.scale(-1e-4)), k_steps)
    opt_init, opt_update = ms_opt.gradient_transformation()

    # Put the training in one function, to check that the update is indeed
    # jittable.
    def train_step(data, opt_state, params):
      grad = jax.grad(loss_apply)(params, data)
      updates, opt_state = opt_update(grad, opt_state, params)
      return updates, opt_state

    opt_state = opt_init(params)

    prev_loss = loss_apply(params, data)
    for idx in range(5 * k_steps):
      updates, opt_state = self.variant(train_step)(data, opt_state, params)
      new_params = update.apply_updates(params, updates)
      new_loss = loss_apply(new_params, data)
      if idx % k_steps < k_steps - 1:
        # The parameters should not have changed and the loss should be
        # constant.
        jax.tree_util.tree_map(
            np.testing.assert_array_equal, new_params, params)
        np.testing.assert_equal(new_loss, prev_loss)
        self.assertFalse(ms_opt.has_updated(opt_state))
      else:
        # This is a step where parameters should actually have been updated, and
        # the loss should accordingly go down.
        np.testing.assert_array_less(new_loss, prev_loss)
        prev_loss = new_loss
        self.assertTrue(ms_opt.has_updated(opt_state))
      params = new_params

  def test_multi_steps_every_k_schedule(self):
    # Test a non-trivial schedule which varies over time.
    ms_opt = wrappers.MultiSteps(
        alias.sgd(1e-4), lambda grad_step: jnp.where(grad_step < 2, 1, 3))
    opt_init, opt_update = ms_opt.gradient_transformation()
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)
    grad = dict(a=jnp.zeros([]))
    self.assertFalse(ms_opt.has_updated(opt_state))
    # First two steps have 1 mini-step per update.
    for _ in range(2):
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertTrue(ms_opt.has_updated(opt_state))
    # Subsequently, mini-steps should have 3 mini-steps per update.
    for _ in range(5):
      for _ in range(2):
        _, opt_state = opt_update(grad, opt_state, params)
        self.assertFalse(ms_opt.has_updated(opt_state))
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertTrue(ms_opt.has_updated(opt_state))

  def test_multi_steps_computes_mean(self):
    k_steps = 4
    ms_opt = wrappers.MultiSteps(
        transform.scale(1.0), k_steps, use_grad_mean=True)
    opt_init, opt_update = ms_opt.gradient_transformation()
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)
    grads = [dict(a=jnp.ones([]) * i) for i in [1, 2, 3, 4]]
    self.assertFalse(ms_opt.has_updated(opt_state))

    # First 3 steps don't update.
    for grad in grads[:-1]:
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertFalse(ms_opt.has_updated(opt_state))

    # Actual update.
    new_params, opt_state = opt_update(grads[-1], opt_state, params)
    self.assertTrue(ms_opt.has_updated(opt_state))
    np.testing.assert_array_equal(new_params['a'], 2.5)

  def test_skip_not_finite(self):
    step = jnp.zeros([], dtype=jnp.int32)

    with self.subTest('test_pos_inf'):
      should_skip, skip_state = wrappers.skip_not_finite(
          [jnp.array(float('inf')), jnp.zeros([])], step, None)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_neg_inf'):
      should_skip, skip_state = wrappers.skip_not_finite(
          [jnp.array(-float('inf')), jnp.zeros([])], step, None)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_nan'):
      should_skip, skip_state = wrappers.skip_not_finite(
          [jnp.array(float('nan')), jnp.zeros([])], step, None)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_finite'):
      should_skip, skip_state = wrappers.skip_not_finite(
          [jnp.array(11.), jnp.zeros([])], step, None)
      self.assertFalse(bool(should_skip))
      self.assertFalse(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 0)

  def test_skip_large_updates(self):
    step = jnp.zeros([], dtype=jnp.int32)

    with self.subTest('test_inf'):
      should_skip, skip_state = wrappers.skip_large_updates(
          [jnp.array(float('inf')), jnp.zeros([])], step, None, 100.)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), float('inf'))

    with self.subTest('test_nan'):
      should_skip, skip_state = wrappers.skip_large_updates(
          [jnp.array(float('nan')), jnp.zeros([])], step, None, 100.)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      # Recall that NaN != NaN.
      norm_squared = float(skip_state['norm_squared'])
      self.assertNotEqual(norm_squared, norm_squared)

    with self.subTest('test_large'):
      should_skip, skip_state = wrappers.skip_large_updates(
          [jnp.array(11.), jnp.zeros([])], step, None, 100.)
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), 121.)

    with self.subTest('test_small'):
      should_skip, skip_state = wrappers.skip_large_updates(
          [jnp.zeros([]), jnp.zeros([])], step, None, 100.)
      self.assertFalse(bool(should_skip))
      self.assertFalse(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), 0.)

  def test_multi_steps_skip_not_finite(self):
    k_steps = 2
    ms_opt = wrappers.MultiSteps(
        alias.sgd(1.), k_steps, should_skip_update_fn=wrappers.skip_not_finite)
    opt_init, opt_update = ms_opt.gradient_transformation()
    opt_init = jax.jit(opt_init)
    opt_update = jax.jit(opt_update)
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)

    with self.subTest('test_good_updates'):
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 1)
      params = update.apply_updates(params, updates)
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], -jnp.ones([]))

    with self.subTest('test_inf_updates'):
      updates, opt_state = opt_update(
          dict(a=jnp.array(float('inf'))), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)  # No increase in mini_step
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], -jnp.ones([]))

    with self.subTest('test_nan_updates'):
      updates, opt_state = opt_update(
          dict(a=jnp.full([], float('nan'))), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)  # No increase in mini_step
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], -jnp.ones([]))

    with self.subTest('test_final_good_updates'):
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 1)
      params = update.apply_updates(params, updates)
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], -jnp.full([], 2.))


class MaskedTest(chex.TestCase):
  """Tests for the masked wrapper."""

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd, False),
      ('stateful_sgd', _build_stateful_sgd, False),
      ('sgd_w_mask_fn', _build_sgd, True),
      ('stateful_sgd_w_mask_fn', _build_stateful_sgd, True),
  )
  def test_masked(self, opt_builder, use_fn):
    mask = {'a': True,
            'b': [False, True],
            'c': {'d': True, 'e': (False, True)}}
    mask_arg = lambda _: mask if use_fn else mask
    params = {'a': 1., 'b': [2., 3.], 'c': {'d': 4., 'e': (5., 6.)}}
    params = jax.tree_util.tree_map(jnp.asarray, params)
    input_updates = jax.tree_util.tree_map(lambda x: x/10., params)

    # Negate the updates wherever the mask is True
    def masked_negate(updates):
      return jax.tree_util.tree_map(
          lambda upd, m: -upd if m else upd, updates, mask)
    correct_updates = masked_negate(input_updates)

    init_fn, update_fn = wrappers.masked(opt_builder(), mask_arg)
    update_fn = self.variant(update_fn)
    state = self.variant(init_fn)(params)
    updates, state = update_fn(input_updates, state, params)
    chex.assert_tree_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = masked_negate(correct_updates)
    updates, state = update_fn(updates, state)
    chex.assert_tree_all_close(updates, correct_updates)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
  )
  def test_prefix_mask(self, opt_builder):
    """Test when the mask is a prefix of the updates PyTree."""
    mask = {'a': True, 'b': False, 'c': {'d': False, 'e': True}}
    params = {'a': 1., 'b': {'f': 2.}, 'c': {'d': 3., 'e': ([4., 5.], 6.)}}
    params = jax.tree_util.tree_map(jnp.asarray, params)
    input_updates = jax.tree_util.tree_map(lambda x: x/10., params)

    # Negate the updates wherever the mask (or mask parent) is True
    def _masked_sgd_on_updates(m, upd):
      return jax.tree_util.tree_map(lambda x: -x, upd) if m else upd
    correct_updates = jax.tree_util.tree_map(
        _masked_sgd_on_updates, mask, input_updates)

    init_fn, update_fn = wrappers.masked(opt_builder(), mask)
    update_fn = self.variant(update_fn)
    state = self.variant(init_fn)(params)
    updates, state = update_fn(input_updates, state, params)
    chex.assert_tree_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = jax.tree_util.tree_map(
        _masked_sgd_on_updates, mask, correct_updates)
    updates, state = update_fn(updates, state)
    chex.assert_tree_all_close(updates, correct_updates)

  @chex.all_variants
  def test_update_requires_params(self):
    weight_decay = 0.1
    mask = {'a': True,
            'b': [False, True],
            'c': {'d': True, 'e': (False, True)}}
    params = {'a': 1., 'b': [2., 3.], 'c': {'d': 4., 'e': (5., 6.)}}
    params = jax.tree_util.tree_map(jnp.asarray, params)
    input_updates = jax.tree_util.tree_map(lambda x: x/10., params)

    correct_updates = jax.tree_util.tree_map(
        lambda m, u, p: u + weight_decay * p if m else u,
        mask, input_updates, params)

    init_fn, update_fn = wrappers.masked(
        transform.additive_weight_decay(weight_decay), mask)
    update_fn = self.variant(update_fn)

    state = self.variant(init_fn)(params)
    updates, state = update_fn(input_updates, state, params)
    chex.assert_tree_all_close(updates, correct_updates)

    params = update.apply_updates(params, updates)

    # Test repeated application
    new_correct_updates = jax.tree_util.tree_map(
        lambda m, u, p: u + weight_decay * p if m else u,
        mask, correct_updates, params)
    updates, state = update_fn(correct_updates, state, params)
    chex.assert_tree_all_close(updates, new_correct_updates)

  @parameterized.parameters(list, tuple, dict)
  def test_empty(self, container):
    init_fn, update_fn = wrappers.masked(_build_sgd(), container())
    update_fn(container(), init_fn(container()))

  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True))
  def test_tree_mismatch_fails(self, extra_key_in_mask, use_fn):
    mask = {'a': True,
            'b': [False, True],
            'c': {'d': True, 'e': (False, True)}}
    mask_arg = lambda _: mask if use_fn else mask
    params = {'a': 1., 'b': [2., 3.], 'c': {'d': 4., 'e': (5., 6.)}}
    params = jax.tree_util.tree_map(jnp.asarray, params)

    if extra_key_in_mask:
      mask['c']['extra'] = True
    else:
      params['c']['extra'] = 7

    init_fn = wrappers.masked(_build_sgd(), mask_arg)[0]
    with self.assertRaises(ValueError):
      init_fn(params)

  @chex.all_variants
  def test_mask_fn(self):
    params = {'a': jnp.ones((1, 2)), 'b': (jnp.ones((1,)), np.ones((1, 2, 3)))}
    mask_fn = lambda p: jax.tree_util.tree_map(lambda x: x.ndim > 1, p)
    init_fn, update_fn = wrappers.masked(transform.add_decayed_weights(0.1),
                                         mask_fn)
    update_fn = self.variant(update_fn)

    state = self.variant(init_fn)(params)
    grads = jax.tree_util.tree_map(lambda x: x*2, params)
    updates, state = update_fn(grads, state, params)
    np.testing.assert_allclose(updates['a'], grads['a'] + 0.1*params['a'])
    np.testing.assert_allclose(updates['b'][0], grads['b'][0])
    np.testing.assert_allclose(updates['b'][1],
                               grads['b'][1] + 0.1*params['b'][1])

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
  )
  def test_nested_mask(self, opt_builder):
    # https://github.com/deepmind/optax/issues/271
    params = {'linear_1': {'w': jnp.zeros((1, 1)), 'b': jnp.zeros(1)},
              'linear_2': {'w': jnp.zeros((1, 2)), 'b': jnp.zeros(2)},
              'linear_3': {'w': jnp.zeros((2, 3)), 'b': jnp.zeros(3)}}

    outer_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim > 1, p)
    inner_mask = jax.tree_util.tree_map(lambda _: True, params)
    inner_mask['linear_2'] = False

    inner = wrappers.masked(opt_builder(), inner_mask)
    init_fn, update_fn = wrappers.masked(inner, outer_mask)

    input_updates = jax.tree_util.tree_map(jnp.ones_like, params)
    correct_updates = copy.deepcopy(input_updates)
    correct_updates['linear_1']['w'] *= -1.0
    correct_updates['linear_3']['w'] *= -1.0

    state = self.variant(init_fn)(params)
    updates, state = self.variant(update_fn)(input_updates, state, params)
    chex.assert_trees_all_close(updates, correct_updates)

  @chex.all_variants
  def test_masked_state_structure(self):
    # https://github.com/deepmind/optax/issues/271
    params = {'a': [jnp.ones(1), (jnp.ones(2), jnp.ones(3))],
              'b': {'c': jnp.ones(4), 'd': jnp.ones(5)}}
    mask = {'a': [True, (True, False)], 'b': False}
    tx = wrappers.masked(_build_stateful_sgd(), mask)
    trace = self.variant(tx.init)(params).inner_state[0].trace
    expected_trace = {
        'a': [jnp.zeros(1), (jnp.zeros(2), wrappers.MaskedNode())],
        'b': wrappers.MaskedNode()
    }
    chex.assert_tree_all_equal_structs(trace, expected_trace)

  def test_masked_state_is_compatible_with_deepmind_tree(self):
    """Checks that the masked state is compatible with deepmind/tree.

    DeepMind's tree library and `jax.tree_util` have slightly different
    behavior: jax treats `None`s as tree nodes without children while
    deepmind/tree treats them as leaves with `None` values. This has led to bugs
    when users used deepmind/tree to manipulate masked optimizer states.

    This test ensures that masked parts of the optimizer state are also ignored
    by deepmind/tree.
    """
    params = {
        'a': [jnp.ones(1), (jnp.ones(2), jnp.ones(3))],
        'b': [jnp.ones(4)]
    }
    mask = {'a': [True, (True, False)], 'b': False}
    opt_init, _ = wrappers.masked(_build_stateful_sgd(), mask)
    state = opt_init(params)
    chex.assert_trees_all_equal(tree.map_structure(np.array, state), state)


class MaybeUpdateTest(chex.TestCase):
  """Tests for the maybe_update wrapper."""

  NUM_STEPS = 3

  @chex.all_variants
  def test_stateless_inner(self):
    params = jnp.zeros([])
    grads = jnp.ones([])

    def should_update(step):
      return step < MaybeUpdateTest.NUM_STEPS

    opt = wrappers.maybe_update(transform.scale(2.), should_update)
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(MaybeUpdateTest.NUM_STEPS):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 2.)
    # Further updates stop calling the inner optimiser.
    for _ in range(5):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 1.)

  @chex.all_variants
  def test_statefull_inner(self):
    params = jnp.zeros([])
    grads_with_nan = jnp.array(float('nan'))
    grads = jnp.ones([])

    def should_update(step):
      return step < MaybeUpdateTest.NUM_STEPS

    opt = wrappers.maybe_update(constrain.zero_nans(), should_update)
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(MaybeUpdateTest.NUM_STEPS - 1):
      updates, state = update_fn(grads_with_nan, state)
      self.assertEqual(updates, 0.)
      self.assertEqual(state.inner_state.found_nan, True)
    updates, state = update_fn(grads, state)
    self.assertEqual(updates, 1.)
    self.assertEqual(state.inner_state.found_nan, False)
    # Further updates stop calling the inner optimiser.
    for _ in range(5):
      updates, state = update_fn(grads_with_nan, state)
      # Warning: do not use assertEqual with a NaN as NaN == NaN returns False.
      self.assertTrue(jnp.isnan(updates))
      # Inner state is not be updated.
      self.assertEqual(state.inner_state.found_nan, False)


if __name__ == '__main__':
  absltest.main()
