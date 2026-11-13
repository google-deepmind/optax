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
"""Tests for methods in `optax.transforms._layouts.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import optax
from optax._src import alias
from optax._src import test_utils
from optax._src import update
from optax.transforms import _layouts


class LayoutsTest(parameterized.TestCase):

  def test_flatten(self):
    def init_params():
      return (jnp.array(2.0), jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

    per_step_updates = (
        jnp.array(1.0),
        jnp.array([500.0, 5.0]),
        jnp.array([300.0, 3.0]),
    )

    # First calculate new params without flattening
    optax_sgd_params = init_params()
    sgd = alias.sgd(1e-2, 0.0)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, _ = sgd.update(per_step_updates, state_sgd)
    sgd_params_no_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # And now calculate new params with flattening
    optax_sgd_params = init_params()
    sgd = _layouts.flatten(sgd)

    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, _ = sgd.update(per_step_updates, state_sgd)
    sgd_params_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # Test that both give the same result
    test_utils.assert_trees_all_close(
        sgd_params_no_flatten, sgd_params_flatten, atol=1e-7, rtol=1e-7
    )

  @parameterized.product(
      params=({}, [], (None, {}), {'w': optax.MaskedNode()}),
      use_jit=(False, True),
  )
  def test_flatten_empty_tree(self, params, use_jit):
    inner = optax.sgd(optax.linear_schedule(0.1, 0.0, 4), momentum=0.9)
    tx = optax.flatten(inner)
    init_fn = jax.jit(tx.init) if use_jit else tx.init
    update_fn = jax.jit(tx.update) if use_jit else tx.update
    state = init_fn(params)
    flat_updates = jnp.zeros((0,))
    expected_state = inner.init(flat_updates)

    for _ in range(3):
      updates, state = update_fn(params, state, params)
      _, expected_state = inner.update(flat_updates, expected_state)
      self.assertEqual(jax.tree.structure(updates), jax.tree.structure(params))
      self.assertEmpty(jax.tree.leaves(updates))
      test_utils.assert_trees_all_equal(state, expected_state)

  @parameterized.product(
      mask=({'w': False, 'b': False}, {'w': True, 'b': False}, True),
      use_jit=(False, True),
  )
  def test_flatten_masked(self, mask, use_jit):
    params = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array(3.0)}
    grads = jax.tree.map(jnp.ones_like, params)
    inner = optax.sgd(optax.linear_schedule(0.1, 0.0, 4), momentum=0.9)
    tx = optax.masked(optax.flatten(inner), mask)
    reference = optax.masked(inner, mask)
    init_fn = jax.jit(tx.init) if use_jit else tx.init
    update_fn = jax.jit(tx.update) if use_jit else tx.update
    state = init_fn(params)
    reference_state = reference.init(params)

    for _ in range(3):
      updates, state = update_fn(grads, state)
      expected, reference_state = reference.update(grads, reference_state)
      test_utils.assert_trees_all_close(updates, expected)

  @parameterized.parameters(False, True)
  def test_flatten_unused_partition(self, use_jit):
    params = {'w': jnp.array([1.0, 2.0])}
    grads = {'w': jnp.array([2.0, -1.0])}
    active = optax.sgd(0.1, momentum=0.9)
    tx = optax.partition(
        {'active': active, 'unused': optax.flatten(optax.adam(0.01))},
        {'w': 'active'},
    )
    init_fn = jax.jit(tx.init) if use_jit else tx.init
    update_fn = jax.jit(tx.update) if use_jit else tx.update
    state = init_fn(params)
    reference_state = active.init(params)

    for _ in range(3):
      updates, state = update_fn(grads, state, params)
      expected, reference_state = active.update(grads, reference_state, params)
      test_utils.assert_trees_all_close(updates, expected)


if __name__ == '__main__':
  absltest.main()
