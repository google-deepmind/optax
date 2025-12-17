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
"""Tests for methods in `schedule.py`."""

import functools
import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import alias
from optax._src import alias_test
from optax._src import base
from optax._src import test_utils
from optax._src import update
from optax.schedules import _schedule


class ConstantTest(absltest.TestCase):

  def test_constant(self):
    """Check constant schedule."""
    # Get schedule function.
    const_value = 10
    num_steps = 15
    schedule_fn = jax.jit(_schedule.constant_schedule(const_value))
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array([const_value] * num_steps, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )


class PolynomialTest(absltest.TestCase):

  def test_linear(self):
    """Check linear schedule."""
    # Get schedule function.
    schedule_fn = jax.jit(
        _schedule.polynomial_schedule(
            init_value=10.0, end_value=20.0, power=1, transition_steps=10
        )
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(list(range(10, 20)) + [20] * 5, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  def test_zero_steps_schedule(self):
    # Get schedule function.
    initial_value = 10.0
    end_value = 20.0

    for num_steps in [-1, 0]:
      schedule_fn = jax.jit(
          _schedule.polynomial_schedule(
              init_value=initial_value,
              end_value=end_value,
              power=1,
              transition_steps=num_steps,
          )
      )
      for count in range(15):
        np.testing.assert_allclose(schedule_fn(count), initial_value)

  def test_nonlinear(self):
    """Check non-linear (quadratic) schedule."""
    # Get schedule function.
    schedule_fn = jax.jit(
        _schedule.polynomial_schedule(
            init_value=25.0, end_value=10.0, power=2, transition_steps=10
        )
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(
        [10.0 + 15.0 * (1.0 - n / 10) ** 2 for n in range(10)] + [10] * 5,
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  def test_with_decay_begin(self):
    """Check quadratic schedule with non-zero schedule begin."""
    # Get schedule function.
    schedule_fn = jax.jit(
        _schedule.polynomial_schedule(
            init_value=30.0,
            end_value=10.0,
            power=2,
            transition_steps=10,
            transition_begin=4,
        )
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(20):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array(
        [30.0] * 4
        + [10.0 + 20.0 * (1.0 - n / 10) ** 2 for n in range(10)]
        + [10] * 6,
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )


class PiecewiseConstantTest(absltest.TestCase):

  def test_positive(self):
    """Check piecewise constant schedule of positive values."""
    # Get schedule function.
    schedule_fn = jax.jit(
        _schedule.piecewise_constant_schedule(0.1, {3: 2.0, 6: 0.5})
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  def test_negative(self):
    """Check piecewise constant schedule of negative values."""
    # Get schedule function.
    schedule_fn = jax.jit(
        _schedule.piecewise_constant_schedule(-0.1, {3: 2.0, 6: 0.5})
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_vals = -1 * np.array(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    )
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )


class ExponentialTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_constant_schedule(self, staircase):
    """Checks constant schedule for exponential decay schedule."""
    num_steps = 15
    # Get schedule function.
    init_value = 1.0
    schedule_fn = jax.jit(
        _schedule.exponential_decay(
            init_value=init_value,
            transition_steps=num_steps,
            decay_rate=1.0,
            staircase=staircase,
        )
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      generated_vals.append(schedule_fn(count))
    expected_vals = np.array([init_value] * num_steps, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  @parameterized.parameters(False, True)
  def test_nonvalid_transition_steps(self, staircase):
    """Checks nonvalid decay steps results in a constant schedule."""
    init_value = 1.0
    for transition_steps in [-1, 0]:
      schedule_fn = jax.jit(
          _schedule.exponential_decay(
              init_value=init_value,
              transition_steps=transition_steps,
              decay_rate=1.0,
              staircase=staircase,
          )
      )
      for count in range(15):
        np.testing.assert_allclose(schedule_fn(count), init_value)

  @parameterized.parameters(False, True)
  def test_nonvalid_decay_rate(self, staircase):
    """Checks nonvalid decay steps results in a constant schedule."""
    init_value = 1.0
    schedule_fn = jax.jit(
        _schedule.exponential_decay(
            init_value=init_value,
            transition_steps=2,
            decay_rate=0.0,
            staircase=staircase,
        )
    )
    for count in range(15):
      np.testing.assert_allclose(schedule_fn(count), init_value)

  @parameterized.parameters((False, 0), (True, 0), (False, 5), (True, 5))
  def test_exponential(self, staircase, transition_begin):
    """Checks non-linear (quadratic) schedule."""
    # Get schedule function.
    init_value = 1.0
    num_steps = 15
    transition_steps = 2
    decay_rate = 2.0
    schedule_fn = jax.jit(
        _schedule.exponential_decay(
            init_value=init_value,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            transition_begin=transition_begin,
            staircase=staircase,
        )
    )

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
        [init_value] * transition_begin
        + [
            init_value * np.power(decay_rate, _staircased(count))
            for count in range(num_steps)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  @parameterized.parameters(
      (0.2, 0.1, False),
      (1.0, 0.1, False),
      (2.0, 3.0, False),
      (0.2, 0.1, True),
      (1.0, 0.1, True),
      (2.0, 3.0, True),
  )
  def test_end_value_with_staircase(self, decay_rate, end_value, staircase):
    # Get schedule function.
    init_value = 1.0
    num_steps = 11
    transition_steps = 2
    transition_begin = 3
    schedule_fn = jax.jit(
        _schedule.exponential_decay(
            init_value=init_value,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            transition_begin=transition_begin,
            staircase=staircase,
            end_value=end_value,
        )
    )

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
        [init_value] * transition_begin
        + [
            init_value * np.power(decay_rate, _staircased(count))
            for count in range(num_steps)
        ],
        dtype=np.float32,
    )

    if decay_rate < 1.0:
      expected_vals = np.maximum(expected_vals, end_value)
    else:
      expected_vals = np.minimum(expected_vals, end_value)

    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )

  def test_immutable_count(self):
    """Checks constant schedule for exponential decay schedule."""
    num_steps = 5
    # Get schedule function.
    init_value = 32.0
    schedule_fn = jax.jit(
        _schedule.exponential_decay(
            init_value=init_value, transition_steps=1, decay_rate=0.5
        )
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(num_steps):
      generated_vals.append(schedule_fn(count))
    expected_vals = np.array([32, 16, 8, 4, 2], dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3
    )


class CosineDecayTest(absltest.TestCase):

  def test_decay_count_smaller_count(self):
    """Check cosine schedule decay for the entire training schedule."""
    initial_value = 0.1
    schedule_fn = jax.jit(
        _schedule.cosine_decay_schedule(initial_value, 10, 0.0)
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(10):
      # Compute next value.
      generated_vals.append(schedule_fn(count))
    # Test output.
    expected_multipliers = np.array(
        0.5
        + 0.5
        * np.cos(
            np.pi * np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )
    )
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals),
        atol=1e-3,
    )

  def test_decay_count_greater_count(self):
    """Check cosine schedule decay for a part of the training schedule."""
    initial_value = 0.1
    schedule_fn = jax.jit(
        _schedule.cosine_decay_schedule(initial_value, 5, 0.0)
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(12):
      # Compute next value.
      generated_vals.append(schedule_fn(count))

    # Test output.
    expected_multipliers = np.array(
        0.5
        + 0.5
        * np.cos(
            np.pi
            * np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        )
    )
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals),
        atol=1e-3,
    )

  def test_decay_count_greater_count_with_alpha(self):
    """Check cosine schedule decay for a part of the training schedule."""
    # Get schedule function.
    initial_value = 0.1
    schedule_fn = jax.jit(
        _schedule.cosine_decay_schedule(initial_value, 5, 0.1)
    )
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(12):
      # Compute next value.
      generated_vals.append(schedule_fn(count))

    # Test output.
    expected_multipliers = np.array(
        0.5
        + 0.5
        * np.cos(
            np.pi
            * np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        )
    )
    expected_multipliers = 0.9 * expected_multipliers + 0.1
    np.testing.assert_allclose(
        initial_value * expected_multipliers,
        np.array(generated_vals),
        atol=1e-3,
    )

  def test_with_exponent(self):
    """Check cosine schedule decay with exponent on."""
    schedule_fn = jax.jit(
        _schedule.cosine_decay_schedule(
            init_value=0.1, decay_steps=100, alpha=0.0, exponent=2
        )
    )
    output = schedule_fn(np.array([0, 10, 50, 75, 100]))
    np.testing.assert_allclose(
        output,
        np.array([0.1, 0.09516553580760956, 0.025, 0.0021446612663567066, 0.0]),
        rtol=1e-6,
        atol=1e-8,
    )

  def test_with_giant_int_steps(self):
    """Check cosine schedule decay with decay_steps not fitting into int32."""
    schedule_fn = jax.jit(
        _schedule.cosine_decay_schedule(
            init_value=1000.0, decay_steps=int(1e10), alpha=0.0, exponent=1
        )
    )
    output = schedule_fn(int(1e9))
    np.testing.assert_allclose(output, 975.52826)


class WarmupCosineDecayTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('with end value', 10, 0.5, 1e-4),
      ('without end value', 5, 3, 0.0),
  )
  def test_limits(self, init_value, peak_value, end_value):
    """Check cosine schedule decay for the entire training schedule."""
    schedule_fn = jax.jit(
        _schedule.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=100,
            decay_steps=1000,
            end_value=end_value,
        )
    )

    np.testing.assert_allclose(init_value, schedule_fn(0))
    np.testing.assert_allclose(peak_value, schedule_fn(100))
    np.testing.assert_allclose(end_value, schedule_fn(1000), rtol=1e-3)

  def test_with_exponent(self):
    """Check that we get correct results when running with exponent on."""
    schedule_fn = jax.jit(
        _schedule.warmup_cosine_decay_schedule(
            init_value=0.2,
            peak_value=1.21,
            end_value=-3.0,
            warmup_steps=50,
            decay_steps=100,
            exponent=2,
        )
    )
    output = schedule_fn(np.array([0, 10, 50, 75, 100]))
    np.testing.assert_allclose(
        output,
        np.array([
            0.20000004768371582,
            0.4020000100135803,
            1.2100000381469727,
            -1.947500228881836,
            -3.000000238418579,
        ]),
        rtol=1e-6,
        atol=1e-8,
    )

  def test_zero_peak_value(self):
    """Check that we get correct results when running with zero peak value."""
    schedule_fn = jax.jit(
        _schedule.warmup_cosine_decay_schedule(
            init_value=0.2,
            peak_value=0,
            end_value=-3.0,
            warmup_steps=50,
            decay_steps=100,
            exponent=2,
        )
    )
    output = schedule_fn(np.array([0, 10, 50, 75, 100]))
    np.testing.assert_allclose(
        output, np.array([0.2, 0.16, 0.0, 0.0, 0.0]), rtol=1e-6, atol=1e-8
    )


class SGDRTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('with step decay', 1.6, 0.8, 0.4),
      ('without step_decay', 1.6, 1.6, 1.6),
  )
  def test_limits(self, lr0, lr1, lr2):
    """Check cosine schedule decay for the entire training schedule."""
    lr_kwargs = []
    for step, lr in zip([2e3, 3e3, 5e3], [lr0, lr1, lr2]):
      lr_kwargs += [{
          'decay_steps': int(step),
          'peak_value': lr,
          'init_value': 0,
          'end_value': 0.0,
          'warmup_steps': 500,
      }]
    schedule_fn = jax.jit(_schedule.sgdr_schedule(lr_kwargs))
    np.testing.assert_allclose(lr0, schedule_fn(500))
    np.testing.assert_allclose(lr1, schedule_fn(2500))
    np.testing.assert_allclose(lr2, schedule_fn(5500))


class PiecewiseInterpolateTest(absltest.TestCase):

  def test_linear_piecewise(self):
    schedule_fn = jax.jit(
        _schedule.piecewise_interpolate_schedule(
            'linear', 200.0, {5: 1.5, 10: 0.25}
        )
    )
    generated_vals = [schedule_fn(step) for step in range(13)]
    expected_vals = [
        200.0,
        220.0,
        240.0,
        260.0,
        280.0,
        300.0,
        255.0,
        210.0,
        165.0,
        120.0,
        75.0,
        75.0,
        75.0,
    ]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_cos_piecewise(self):
    schedule_fn = jax.jit(
        _schedule.piecewise_interpolate_schedule(
            'cosine', 400.0, {5: 1.2, 3: 0.6, 7: 1.0}
        )
    )
    generated_vals = [schedule_fn(step) for step in range(9)]
    expected_vals = [
        400.0,
        360.0,
        280.0,
        240.0,
        264.0,
        288.0,
        288.0,
        288.0,
        288.0,
    ]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_empty_dict(self):
    schedule_fn = jax.jit(
        _schedule.piecewise_interpolate_schedule('linear', 13.0, {})
    )
    generated_vals = [schedule_fn(step) for step in range(5)]
    expected_vals = [13.0, 13.0, 13.0, 13.0, 13.0]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_no_dict(self):
    schedule_fn = jax.jit(
        _schedule.piecewise_interpolate_schedule('cosine', 17.0)
    )
    generated_vals = [schedule_fn(step) for step in range(3)]
    expected_vals = [17.0, 17.0, 17.0]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_invalid_type(self):
    # pytype: disable=wrong-arg-types
    with self.assertRaises(ValueError):
      _schedule.piecewise_interpolate_schedule('linar', 13.0)
    with self.assertRaises(ValueError):
      _schedule.piecewise_interpolate_schedule('', 13.0, {5: 3.0})
    with self.assertRaises(ValueError):
      _schedule.piecewise_interpolate_schedule(None, 13.0, {})
    # pytype: enable=wrong-arg-types

  def test_invalid_scale(self):
    with self.assertRaises(ValueError):
      _schedule.piecewise_interpolate_schedule('linear', 13.0, {5: -3})

  def test_zero_interval_size(self):
    sched = _schedule.piecewise_interpolate_schedule(
        'linear', 13.0, {0: 2.0, 5: 3.0}
    )
    self.assertTrue(all(jnp.isfinite(sched(i)) for i in range(6)))
    self.assertAlmostEqual(sched(0), 26.0)  # 13.0 * 2.0
    self.assertAlmostEqual(sched(5), 78.0)  # 26.0 * 3.0


class OneCycleTest(absltest.TestCase):

  def test_linear(self):
    schedule_fn = jax.jit(
        _schedule.linear_onecycle_schedule(
            transition_steps=10,
            peak_value=1000,
            pct_start=0.3,
            pct_final=0.7,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    )

    generated_vals = [schedule_fn(step) for step in range(12)]
    expected_vals = [
        100.0,
        400.0,
        700.0,
        1000.0,
        775.0,
        550.0,
        325.0,
        100.0,
        67.0,
        34.0,
        1.0,
        1.0,
    ]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_cosine(self):
    schedule_fn = jax.jit(
        _schedule.cosine_onecycle_schedule(
            transition_steps=5,
            peak_value=1000.0,
            pct_start=0.4,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    )

    generated_vals = [schedule_fn(step) for step in range(7)]
    expected_vals = [100.0, 550.0, 1000.0, 750.25, 250.75, 1.0, 1.0]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_nonpositive_transition_steps(self):
    with self.assertRaises(ValueError):
      _schedule.cosine_onecycle_schedule(transition_steps=0, peak_value=5.0)
    with self.assertRaises(ValueError):
      _schedule.linear_onecycle_schedule(transition_steps=0, peak_value=5.0)


class ScheduleAsLearningRateTest(parameterized.TestCase):
  @parameterized.product(alias_test._OPTIMIZERS_UNDER_TEST)
  def test_optimization(self, opt_name, opt_kwargs):
    opt_setup = getattr(alias, opt_name)

    # check if the optimizer is annotated to accept a learning rate schedule
    opt_setup_signature = inspect.signature(opt_setup)
    lr_arg = opt_setup_signature.parameters.get('learning_rate', None)
    if lr_arg is None or lr_arg.annotation not in (base.ScalarOrSchedule,
                                                   base.Schedule):
      self.skipTest('Optimizer doesn\'t accept a learning schedule.')

    opt_kwargs_with_schedule = dict(
        opt_kwargs, learning_rate=_schedule.constant_schedule(1e-3))
    opt_kwargs_with_lr = dict(opt_kwargs, learning_rate=1e-3)

    # setup the optimizer and run on a test function
    opt_with_schedule = opt_setup(**opt_kwargs_with_schedule)
    opt_with_lr = opt_setup(**opt_kwargs_with_lr)
    initial_params, _, objective = alias_test._setup_parabola(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(opt, params, state):
      value, updates = jax.value_and_grad(objective)(params)
      # Complex gradients need to be conjugated before being added to parameters
      # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
      updates = jax.tree.map(lambda x: x.conj(), updates)
      if opt_name == 'polyak_sgd':
        update_kwargs = {'value': value}
      else:
        update_kwargs = {}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params_with_schedule, params_with_lr = initial_params, initial_params
    state_with_schedule = opt_with_schedule.init(params_with_schedule)
    state_with_lr = opt_with_lr.init(params_with_lr)

    with self.subTest('Test that optimizer accepts learning rate schedule as'
                      ' learning_rate'):
      for _ in range(10):
        params_with_schedule, state_with_schedule = step(
            opt_with_schedule, params_with_schedule, state_with_schedule)
        params_with_lr, state_with_lr = step(
            opt_with_lr, params_with_lr, state_with_lr)
      # check if the constant learning rate schedule and the learning rate
      # are equivalent
      test_utils.assert_trees_all_close(params_with_schedule, params_with_lr)

if __name__ == '__main__':
  absltest.main()
