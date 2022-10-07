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
"""Tests for `schedule.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import clipping
from optax._src import schedule
from optax._src import transform
from optax._src import wrappers


class ConstantTest(chex.TestCase):

  @chex.all_variants
  def test_constant(self):
    """Check constant schedule."""
    # Get schedule function.
    const_value = 10
    num_steps = 15
    schedule_fn = self.variant(schedule.constant_schedule(const_value))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array([const_value] * num_steps, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)


class PolynomialTest(chex.TestCase):

  @chex.all_variants
  def test_linear(self):
    """Check linear schedule."""
    # Get schedule function.
    schedule_fn = self.variant(
        schedule.polynomial_schedule(
            init_value=10., end_value=20., power=1, transition_steps=10))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(list(range(10, 20)) + [20] * 5, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_zero_steps_schedule(self):
    # Get schedule function.
    initial_value = 10.
    end_value = 20.

    for num_steps in [-1, 0]:
      schedule_fn = self.variant(
          schedule.polynomial_schedule(
              init_value=initial_value, end_value=end_value,
              power=1, transition_steps=num_steps))
      for count in range(15):
        np.testing.assert_allclose(schedule_fn(count), initial_value)

  @chex.all_variants
  def test_nonlinear(self):
    """Check non-linear (quadratic) schedule."""
    # Get schedule function.
    schedule_fn = self.variant(
        schedule.polynomial_schedule(
            init_value=25., end_value=10., power=2, transition_steps=10))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(
        [10. + 15. * (1. - n / 10)**2 for n in range(10)] + [10] * 5,
        dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_with_decay_begin(self):
    """Check quadratic schedule with non-zero schedule begin."""
    # Get schedule function.
    schedule_fn = self.variant(
        schedule.polynomial_schedule(
            init_value=30., end_value=10., power=2,
            transition_steps=10, transition_begin=4))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(20):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(
        [30.] * 4 + [10. + 20. * (1. - n / 10)**2 for n in range(10)] +
        [10] * 6,
        dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)


class PiecewiseConstantTest(chex.TestCase):

  @chex.all_variants
  def test_positive(self):
    """Check piecewise constant schedule of positive values."""
    # Get schedule function.
    schedule_fn = self.variant(
        schedule.piecewise_constant_schedule(0.1, {3: 2., 6: 0.5}))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_negative(self):
    """Check piecewise constant schedule of negative values."""
    # Get schedule function.
    schedule_fn = self.variant(
        schedule.piecewise_constant_schedule(-0.1, {3: 2., 6: 0.5}))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = -1 * np.array(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)


class ExponentialTest(chex.TestCase):

  @chex.all_variants
  @parameterized.parameters(False, True)
  def test_constant_schedule(self, staircase):
    """Checks constant schedule for exponential decay schedule."""
    num_steps = 15
    # Get schedule function.
    init_value = 1.
    schedule_fn = self.variant(
        schedule.exponential_decay(
            init_value=init_value, transition_steps=num_steps,
            decay_rate=1., staircase=staircase))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      generated_vals.append(schedule_fn(count))
    expected_vals = np.array([init_value] * num_steps, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  @parameterized.parameters(False, True)
  def test_nonvalid_transition_steps(self, staircase):
    """Checks nonvalid decay steps results in a constant schedule."""
    init_value = 1.
    for transition_steps in [-1, 0]:
      schedule_fn = self.variant(
          schedule.exponential_decay(
              init_value=init_value, transition_steps=transition_steps,
              decay_rate=1., staircase=staircase))
      for count in range(15):
        np.testing.assert_allclose(schedule_fn(count), init_value)

  @chex.all_variants
  @parameterized.parameters(False, True)
  def test_nonvalid_decay_rate(self, staircase):
    """Checks nonvalid decay steps results in a constant schedule."""
    init_value = 1.
    schedule_fn = self.variant(
        schedule.exponential_decay(
            init_value=init_value, transition_steps=2,
            decay_rate=0., staircase=staircase))
    for count in range(15):
      np.testing.assert_allclose(schedule_fn(count), init_value)

  @chex.all_variants
  @parameterized.parameters((False, 0), (True, 0), (False, 5), (True, 5))
  def test_exponential(self, staircase, transition_begin):
    """Checks non-linear (quadratic) schedule."""
    # Get schedule function.
    init_value = 1.
    num_steps = 15
    transition_steps = 2
    decay_rate = 2.
    schedule_fn = self.variant(
        schedule.exponential_decay(
            init_value=init_value, transition_steps=transition_steps,
            decay_rate=decay_rate, transition_begin=transition_begin,
            staircase=staircase))

    # Test that generated values equal the expected schedule values.
    def _staircased(count):
      p = count / transition_steps
      if staircase:
        p = np.floor(p)
      return p

    generated_vals = []
    for count in range(num_steps + transition_begin):
      generated_vals.append(schedule_fn(count))
    expected_vals = np.array(
        [init_value] * transition_begin + [
            init_value * np.power(decay_rate, _staircased(count))
            for count in range(num_steps)
        ],
        dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  @parameterized.parameters(
      (0.2, 0.1, False), (1.0, 0.1, False), (2.0, 3.0, False),
      (0.2, 0.1, True), (1.0, 0.1, True), (2.0, 3.0, True))
  def test_end_value_with_staircase(self, decay_rate, end_value, staircase):
    # Get schedule function.
    init_value = 1.
    num_steps = 11
    transition_steps = 2
    transition_begin = 3
    schedule_fn = self.variant(
        schedule.exponential_decay(
            init_value=init_value, transition_steps=transition_steps,
            decay_rate=decay_rate, transition_begin=transition_begin,
            staircase=staircase, end_value=end_value))

    # Test that generated values equal the expected schedule values.
    def _staircased(count):
      p = count / transition_steps
      if staircase:
        p = np.floor(p)
      return p

    generated_vals = []
    for count in range(num_steps + transition_begin):
      generated_vals.append(schedule_fn(count))
    expected_vals = np.array(
        [init_value] * transition_begin + [
            init_value * np.power(decay_rate, _staircased(count))
            for count in range(num_steps)
        ],
        dtype=np.float32)

    if decay_rate < 1.0:
      expected_vals = np.maximum(expected_vals, end_value)
    else:
      expected_vals = np.minimum(expected_vals, end_value)

    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_immutable_count(self):
    """Checks constant schedule for exponential decay schedule."""
    num_steps = 5
    # Get schedule function.
    init_value = 32.
    schedule_fn = self.variant(
        schedule.exponential_decay(
            init_value=init_value, transition_steps=1,
            decay_rate=0.5))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      # Jax arrays are read-only in ChexVariantType.WITHOUT_DEVICE.
      immutable_count = jnp.array(count, dtype=jnp.float32)
      generated_vals.append(schedule_fn(immutable_count))
    expected_vals = np.array([32, 16, 8, 4, 2], dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)


class CosineDecayTest(chex.TestCase):

  @chex.all_variants
  def test_decay_count_smaller_count(self):
    """Check cosine schedule decay for the entire training schedule."""
    initial_value = 0.1
    schedule_fn = self.variant(
        schedule.cosine_decay_schedule(initial_value, 10, 0.0))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_multipliers = np.array(
        0.5 + 0.5 * np.cos(
            np.pi * np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_decay_count_greater_count(self):
    """Check cosine schedule decay for a part of the training schedule."""
    initial_value = 0.1
    schedule_fn = self.variant(
        schedule.cosine_decay_schedule(initial_value, 5, 0.0))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(12):
      # Compute next value.
      generated_vals.append(schedule_fn(count))

    # Test output.
    expected_multipliers = np.array(
        0.5 + 0.5 * np.cos(
            np.pi * np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1., 1., 1., 1., 1., 1., 1.])))
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals), atol=1e-3)

  @chex.all_variants
  def test_decay_count_greater_count_with_alpha(self):
    """Check cosine schedule decay for a part of the training schedule."""
    # Get schedule function.
    initial_value = 0.1
    schedule_fn = self.variant(
        schedule.cosine_decay_schedule(initial_value, 5, 0.1))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(12):
      # Compute next value.
      generated_vals.append(schedule_fn(count))

    # Test output.
    expected_multipliers = np.array(
        0.5 + 0.5 * np.cos(
            np.pi * np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1., 1., 1., 1., 1., 1., 1.])))
    expected_multipliers = 0.9 * expected_multipliers + 0.1
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals), atol=1e-3)


class WarmupCosineDecayTest(chex.TestCase):

  @chex.all_variants
  @parameterized.named_parameters(
      ('with end value', 10, 0.5, 1e-4),
      ('without end value', 5, 3, 0.),)
  def test_limits(self, init_value, peak_value, end_value):
    """Check cosine schedule decay for the entire training schedule."""
    schedule_fn = self.variant(schedule.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=100,
        decay_steps=1000,
        end_value=end_value,
    ))

    np.testing.assert_allclose(init_value, schedule_fn(0))
    np.testing.assert_allclose(peak_value, schedule_fn(100))
    np.testing.assert_allclose(end_value, schedule_fn(1000), rtol=1e-3)


class SGDRTest(chex.TestCase):

  @chex.all_variants
  @parameterized.named_parameters(
      ('with step decay', 1.6, 0.8, 0.4),
      ('without step_decay', 1.6, 1.6, 1.6),)
  def test_limits(self, lr0, lr1, lr2):
    """Check cosine schedule decay for the entire training schedule."""
    lr_kwargs = []
    for step, lr in zip([2e3, 3e3, 5e3], [lr0, lr1, lr2]):
      lr_kwargs += [dict(decay_steps=int(step), peak_value=lr,
                         init_value=0, end_value=0.0, warmup_steps=500)]
    schedule_fn = self.variant(schedule.sgdr_schedule(lr_kwargs))
    np.testing.assert_allclose(lr0, schedule_fn(500))
    np.testing.assert_allclose(lr1, schedule_fn(2500))
    np.testing.assert_allclose(lr2, schedule_fn(5500))


class PiecewiseInterpolateTest(chex.TestCase):

  @chex.all_variants
  def test_linear_piecewise(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'linear', 200., {5: 1.5, 10: 0.25}))
    generated_vals = [schedule_fn(step) for step in range(13)]
    expected_vals = [200., 220., 240., 260., 280., 300., 255., 210., 165.,
                     120., 75., 75., 75.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants
  def test_cos_piecewise(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'cosine', 400., {5: 1.2, 3: 0.6, 7: 1.}))
    generated_vals = [schedule_fn(step) for step in range(9)]
    expected_vals = [400., 360., 280., 240., 264., 288., 288., 288., 288.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants
  def test_empty_dict(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'linear', 13., {}))
    generated_vals = [schedule_fn(step) for step in range(5)]
    expected_vals = [13., 13., 13., 13., 13.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants
  def test_no_dict(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'cosine', 17.))
    generated_vals = [schedule_fn(step) for step in range(3)]
    expected_vals = [17., 17., 17.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_invalid_type(self):
    # pytype: disable=wrong-arg-types
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('linar', 13.)
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('', 13., {5: 3.})
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule(None, 13., {})
    # pytype: enable=wrong-arg-types

  def test_invalid_scale(self):
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('linear', 13., {5: -3})


class OneCycleTest(chex.TestCase):

  @chex.all_variants
  def test_linear(self):
    schedule_fn = self.variant(schedule.linear_onecycle_schedule(
        transition_steps=10,
        peak_value=1000,
        pct_start=0.3,
        pct_final=0.7,
        div_factor=10.,
        final_div_factor=100.))

    generated_vals = [schedule_fn(step) for step in range(12)]
    expected_vals = [100., 400., 700., 1000., 775., 550., 325., 100., 67.,
                     34., 1., 1.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants
  def test_cosine(self):
    schedule_fn = self.variant(schedule.cosine_onecycle_schedule(
        transition_steps=5,
        peak_value=1000.,
        pct_start=0.4,
        div_factor=10.,
        final_div_factor=100.))

    generated_vals = [schedule_fn(step) for step in range(7)]
    expected_vals = [100., 550., 1000., 750.25, 250.75, 1., 1.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_nonpositive_transition_steps(self):
    with self.assertRaises(ValueError):
      schedule.cosine_onecycle_schedule(transition_steps=0, peak_value=5.)
    with self.assertRaises(ValueError):
      schedule.linear_onecycle_schedule(transition_steps=0, peak_value=5.)


class InjectHyperparamsTest(chex.TestCase):
  """Tests for the inject_hyperparams wrapper."""

  @chex.all_variants
  def test_updates(self):
    optim = schedule.inject_hyperparams(transform.scale)(  # stateless
        step_size=schedule.piecewise_constant_schedule(
            3.0, {1: 5, 7: 2, 12: 1.5}))

    params = [jnp.zeros([], dtype=jnp.float32)]
    state = self.variant(optim.init)(params)
    update_fn = self.variant(optim.update)
    expected_step_size = [3.0]*2 + [15.0]*6 + [30.0]*5 + [45.0]*3

    grads = [jnp.ones([], dtype=jnp.float32)]
    for i in range(15):
      updates, state = update_fn(grads, state, params=params)
      np.testing.assert_almost_equal(updates[0], expected_step_size[i+1])

  @chex.all_variants
  def test_hyperparams_state(self):
    optim = schedule.inject_hyperparams(transform.trace)(  # stateful
        decay=schedule.piecewise_constant_schedule(
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
    optim = schedule.inject_hyperparams(transform.scale_by_adam)(b1=0., b2=0.)

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
      chex.assert_tree_all_close(updates, grads)

  @chex.all_variants
  def test_overriding_hyperparam(self):
    optim = schedule.inject_hyperparams(clipping.clip_by_global_norm)(0.1)
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
    @functools.partial(schedule.inject_hyperparams, static_args=static_args)
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
    chex.assert_tree_all_close(updates, expected_updates)

  @chex.all_variants
  @parameterized.named_parameters(('one_arg', 'b1'), ('two_arg', ['b1', 'b2']))
  def test_numeric_static_args(self, static_args):
    optim = schedule.inject_hyperparams(
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
    optim = schedule.inject_hyperparams(
        transform.scale_by_adam,
        hyperparam_dtype=hyperparam_dtype)(b1=0.9, b2=0.95)

    params = [jnp.ones((1, 2), dtype=param_dtype),
              jnp.ones(2, dtype=param_dtype),
              jnp.ones((1, 1, 1), dtype=param_dtype)]
    grads = jax.tree_map(lambda x: x.astype(grad_dtype), params)
    state = self.variant(optim.init)(params)
    # Check that the hyperparams are overriden
    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

    _, state = self.variant(optim.update)(grads, state)

    self.assertEqual(state.hyperparams['b1'].dtype, hyperparam_dtype)
    self.assertEqual(state.hyperparams['b2'].dtype, hyperparam_dtype)

  @parameterized.named_parameters(('string', 'lr'), ('list', ['lr']))
  def test_static_args_error(self, static_args):
    with self.assertRaises(ValueError):
      schedule.inject_hyperparams(transform.scale, static_args=static_args)

  @chex.all_variants
  def test_inject_hyperparams_starts_with_step_count_zero(self):
    """Checks that inject_hyperparams uses step count 0 in the first update."""
    # See also: https://github.com/deepmind/optax/issues/415.
    opt = schedule.inject_hyperparams(transform.scale)(lambda count: count)
    params = jnp.zeros(3)
    grads = jnp.array([-1, 0, 1])
    updates, _ = self.variant(opt.update)(grads, opt.init(params))
    np.testing.assert_array_equal(updates, np.zeros(3))


if __name__ == '__main__':
  absltest.main()
