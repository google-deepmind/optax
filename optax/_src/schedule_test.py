# Lint as: python3
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

from absl.testing import absltest
from absl.testing import parameterized

import chex
import numpy as np

from optax._src import schedule


class ConstantTest(chex.TestCase):

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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


class CosineDecayTest(chex.TestCase):

  @chex.all_variants()
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

  @chex.all_variants()
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

  @chex.all_variants()
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


class PiecewiseInterpolateTest(chex.TestCase):

  @chex.all_variants()
  def test_linear_piecewise(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'linear', 200., {5: 1.5, 10: 0.25}))
    generated_vals = [schedule_fn(step) for step in range(13)]
    expected_vals = [200., 220., 240., 260., 280., 300., 255., 210., 165.,
                     120., 75., 75., 75.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants()
  def test_cos_piecewise(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'cosine', 400., {5: 1.2, 3: 0.6, 7: 1.}))
    generated_vals = [schedule_fn(step) for step in range(9)]
    expected_vals = [400., 360., 280., 240., 264., 288., 288., 288., 288.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants()
  def test_empty_dict(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'linear', 13., {}))
    generated_vals = [schedule_fn(step) for step in range(5)]
    expected_vals = [13., 13., 13., 13., 13.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  @chex.all_variants()
  def test_no_dict(self):
    schedule_fn = self.variant(schedule.piecewise_interpolate_schedule(
        'cosine', 17.))
    generated_vals = [schedule_fn(step) for step in range(3)]
    expected_vals = [17., 17., 17.]
    np.testing.assert_allclose(generated_vals, expected_vals, atol=1e-3)

  def test_invalid_type(self):
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('linar', 13.)
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('', 13., {5: 3.})
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule(None, 13., {})  # pytype: disable=wrong-arg-types

  def test_invalid_scale(self):
    with self.assertRaises(ValueError):
      schedule.piecewise_interpolate_schedule('linear', 13., {5: -3})


class OneCycleTest(chex.TestCase):

  @chex.all_variants()
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

  @chex.all_variants()
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


if __name__ == '__main__':
  absltest.main()
