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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from optax.schedules import warmup_cosine_decay_schedule


def _sample(sched, n_steps):
    """Return values at steps [0, 1, ..., n_steps]."""
    steps = jnp.arange(n_steps + 1)
    return jax.vmap(sched)(steps)


class WarmupCosineScheduleTest(parameterized.TestCase):

    def test_warmup_increases_and_reaches_peak(self):
        # Valid setting: 1 <= warmup_steps <= decay_steps
        init, peak, end = 0.0, 1.0, 0.1
        warmup_steps, decay_steps = 5, 25

        sched = warmup_cosine_decay_schedule(
            init_value=init,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end,
        )
        vals = _sample(sched, decay_steps)

        # Warmup strictly increasing
        for t in range(warmup_steps):
            self.assertLess(float(vals[t]), float(vals[t + 1]))

        # Value at warmup boundary ≈ peak
        self.assertAlmostEqual(float(vals[warmup_steps]), peak, places=7)

    def test_decay_nonincreasing_and_hits_end_value(self):
        init, peak, end = 0.0, 2.0, 0.5
        warmup_steps, decay_steps = 4, 20

        sched = warmup_cosine_decay_schedule(
            init_value=init,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end,
        )
        vals = _sample(sched, decay_steps)

        # After warmup, cosine phase should be pointwise nonincreasing
        for t in range(warmup_steps, decay_steps):
            self.assertGreaterEqual(float(vals[t]), float(vals[t + 1]))

        # Final value == end_value (within tolerance)
        self.assertAlmostEqual(float(vals[-1]), end, places=7)

    def test_raises_when_decay_equals_warmup(self):
        # By design, decay_steps must be strictly greater than warmup_steps.
        init, peak = 0.2, 1.2
        warmup_steps = decay_steps = 10
        with self.assertRaises(ValueError):
            _ = warmup_cosine_decay_schedule(
                init_value=init,
                peak_value=peak,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=0.0,
            )



    def test_exponent_changes_curve_shape(self):
        init, peak, end = 0.0, 1.0, 0.0
        warmup_steps, decay_steps = 2, 12

        base = warmup_cosine_decay_schedule(
            init_value=init,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end,
            exponent=1.0,
        )
        steeper = warmup_cosine_decay_schedule(
            init_value=init,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end,
            exponent=2.0,
        )

        vb = _sample(base, decay_steps)
        vs = _sample(steeper, decay_steps)

        # After warmup, exponent=2 should never exceed exponent=1 pointwise.
        for t in range(warmup_steps, decay_steps + 1):
            self.assertLessEqual(float(vs[t]), float(vb[t]) + 1e-12)

    def test_accepts_int_steps_and_returns_float_dtype(self):
        sched = warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1.0,
            warmup_steps=3,
            decay_steps=9,
            end_value=0.0,
        )
        y_scalar = sched(jnp.array(4, dtype=jnp.int32))
        self.assertTrue(jnp.issubdtype(y_scalar.dtype, jnp.floating))

        ys = _sample(sched, 9)
        self.assertEqual(ys.dtype, y_scalar.dtype)

    def test_jit_matches_eager(self):
        init, peak, end = 0.0, 1.0, 0.1
        warmup_steps, decay_steps = 5, 25

        sched = warmup_cosine_decay_schedule(
            init_value=init,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end,
        )
        sched_jit = jax.jit(sched)

        steps = jnp.arange(decay_steps + 1)
        eager = jax.vmap(sched)(steps)
        compiled = jax.vmap(sched_jit)(steps)

        self.assertTrue(jnp.allclose(eager, compiled, rtol=1e-6, atol=1e-8))


if __name__ == "__main__":
    absltest.main()
