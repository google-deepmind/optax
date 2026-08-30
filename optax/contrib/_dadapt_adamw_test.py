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
"""Tests for the D-Adaptation AdamW optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax.contrib import _dadapt_adamw

_WEIGHT_DECAY = 1e-1


def _params():
  return {
      'a': {'w': jnp.ones((2, 3)), 'b': jnp.ones((3,))},
      'c': {'w': jnp.ones((3, 2)), 'b': jnp.ones((2,))},
  }


def _updates(params, **kwargs):
  opt = _dadapt_adamw.dadapt_adamw(learning_rate=1e-1, **kwargs)
  grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
  updates, _ = opt.update(grads, opt.init(params), params)
  return updates


class DAdaptAdamWTest(parameterized.TestCase):

  def test_no_mask_decays_everything(self):
    """An all-True mask must be a no-op relative to passing no mask."""
    params = _params()
    unmasked = _updates(params, weight_decay=_WEIGHT_DECAY)
    for mask in (True, jax.tree.map(lambda _: True, params)):
      masked = _updates(
          params, weight_decay=_WEIGHT_DECAY, weight_decay_mask=mask
      )
      test_utils.assert_trees_all_close(unmasked, masked)

  def test_all_false_mask_disables_decay(self):
    params = _params()
    no_decay = _updates(params, weight_decay=0.0)
    masked = _updates(
        params, weight_decay=_WEIGHT_DECAY, weight_decay_mask=False
    )
    test_utils.assert_trees_all_close(no_decay, masked)

  def test_mask_is_applied_per_leaf(self):
    """Only the leaves selected by the mask should be decayed."""
    params = _params()
    decayed = _updates(params, weight_decay=_WEIGHT_DECAY)
    no_decay = _updates(params, weight_decay=0.0)
    # Skip the biases, a common use case.
    mask = jax.tree.map(lambda p: p.ndim != 1, params)
    masked = _updates(
        params, weight_decay=_WEIGHT_DECAY, weight_decay_mask=mask
    )
    expected = jax.tree.map(
        lambda m, d, nd: d if m else nd, mask, decayed, no_decay
    )
    test_utils.assert_trees_all_close(expected, masked)

  def test_prefix_mask(self):
    """A mask may be a prefix of the params tree."""
    params = _params()
    decayed = _updates(params, weight_decay=_WEIGHT_DECAY)
    no_decay = _updates(params, weight_decay=0.0)
    masked = _updates(
        params,
        weight_decay=_WEIGHT_DECAY,
        weight_decay_mask={'a': True, 'c': False},
    )
    test_utils.assert_trees_all_close(decayed['a'], masked['a'])
    test_utils.assert_trees_all_close(no_decay['c'], masked['c'])

  def test_callable_mask(self):
    params = _params()
    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)
    test_utils.assert_trees_all_close(
        _updates(
            params,
            weight_decay=_WEIGHT_DECAY,
            weight_decay_mask=mask_fn(params),
        ),
        _updates(params, weight_decay=_WEIGHT_DECAY, weight_decay_mask=mask_fn),
    )

  def test_mask_is_jittable(self):
    params = _params()
    opt = _dadapt_adamw.dadapt_adamw(
        learning_rate=1e-1,
        weight_decay=_WEIGHT_DECAY,
        weight_decay_mask=lambda p: jax.tree.map(lambda x: x.ndim != 1, p),
    )
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
    state = opt.init(params)
    jitted, _ = jax.jit(opt.update)(grads, state, params)
    eager, _ = opt.update(grads, state, params)
    test_utils.assert_trees_all_close(eager, jitted)

  def test_zero_grads_take_no_step(self):
    """A zero denominator must not produce a non-finite estimate."""
    params = _params()
    opt = _dadapt_adamw.dadapt_adamw(learning_rate=1e-1)
    grads = jax.tree.map(jnp.zeros_like, params)
    state = opt.init(params)
    updates, state = opt.update(grads, state, params)
    test_utils.assert_trees_all_close(
        updates, jax.tree.map(jnp.zeros_like, params)
    )
    self.assertEqual(state.estim_lr, opt.init(params).estim_lr)
    self.assertEqual(state.numerator_weighted, 0.0)
    self.assertEqual(state.count, 0)
    # ... but the schedule must still have advanced.
    self.assertEqual(state.sched_count, 1)

  def test_zero_grads_do_not_poison_later_steps(self):
    """`estim_lr` is a running maximum, so a NaN would never wash out."""
    params = _params()
    opt = _dadapt_adamw.dadapt_adamw(learning_rate=1.0)
    state = opt.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
    _, state = opt.update(zeros, state, params)
    for _ in range(3):
      updates, state = opt.update(grads, state, params)
      self.assertTrue(jnp.isfinite(state.estim_lr))
      self.assertTrue(
          all(jnp.isfinite(u).all() for u in jax.tree.leaves(updates))
      )

  def test_zero_learning_rate_takes_no_step(self):
    """A zero learning rate must leave the state untouched."""
    params = _params()
    opt = _dadapt_adamw.dadapt_adamw(learning_rate=0.0)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
    state = opt.init(params)
    updates, new_state = opt.update(grads, state, params)
    test_utils.assert_trees_all_close(
        updates, jax.tree.map(jnp.zeros_like, params)
    )
    for field in ('exp_avg', 'exp_avg_sq', 'grad_sum'):
      test_utils.assert_trees_all_close(
          getattr(new_state, field), getattr(state, field)
      )
    self.assertEqual(new_state.estim_lr, state.estim_lr)
    self.assertEqual(new_state.numerator_weighted, state.numerator_weighted)
    self.assertEqual(new_state.count, 0)
    self.assertEqual(new_state.sched_count, 1)

  def test_warmup_from_zero_does_not_wedge(self):
    """The recommended warmup must not freeze the schedule or emit NaN."""
    params = _params()
    warmup = lambda c: jnp.where(c < 2, 0.0, 1e-1)
    opt = _dadapt_adamw.dadapt_adamw(learning_rate=warmup)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
    state = opt.init(params)
    for _ in range(2):
      updates, state = opt.update(grads, state, params)
      test_utils.assert_trees_all_close(
          updates, jax.tree.map(jnp.zeros_like, params)
      )
      self.assertTrue(jnp.isfinite(state.estim_lr))
    self.assertEqual(state.count, 0)
    self.assertEqual(state.sched_count, 2)
    # The third step is the first with a non-zero learning rate, so it must
    # make progress, and must stay finite.
    updates, state = opt.update(grads, state, params)
    self.assertEqual(state.count, 1)
    self.assertTrue(jnp.isfinite(state.estim_lr))
    self.assertTrue(
        all(jnp.isfinite(u).all() for u in jax.tree.leaves(updates))
    )

  def test_guards_do_not_change_the_common_case(self):
    """With a positive lr and non-zero grads, nothing should be masked out."""
    params = _params()
    opt = _dadapt_adamw.dadapt_adamw(learning_rate=1e-1)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), params)
    state = opt.init(params)
    for step in range(1, 4):
      updates, state = opt.update(grads, state, params)
      params = jax.tree.map(lambda p, u: p + u, params, updates)
      self.assertEqual(state.count, step)
      self.assertEqual(state.sched_count, step)
    self.assertTrue(all(jnp.isfinite(p).all() for p in jax.tree.leaves(params)))


if __name__ == '__main__':
  absltest.main()
