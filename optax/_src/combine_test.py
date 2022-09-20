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
"""Tests for `combine.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update


STEPS = 50
LR = 1e-2


class ComposeTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants
  def test_chain(self):
    transformations = [
        transform.scale_by_adam(),
        transform.trace(decay=0, nesterov=False),
        transform.scale(-LR)]

    # Apply updates with chain.
    chain_params = self.init_params
    chained_transforms = combine.chain(*transformations)
    state = chained_transforms.init(chain_params)
    self.assertIsInstance(state, tuple)

    @self.variant
    def update_fn(updates, state):
      return chained_transforms.update(updates, state)

    for _ in range(STEPS):
      updates, state = update_fn(self.per_step_updates, state)
      self.assertIsInstance(state, tuple)
      chain_params = update.apply_updates(chain_params, updates)

    # Manually apply sequence of transformations.
    manual_params = self.init_params
    states = [t.init(manual_params) for t in transformations]
    for _ in range(STEPS):
      updates = self.per_step_updates
      new_states = []
      for t, s in zip(transformations, states):
        updates, state = t.update(updates, s)
        new_states.append(state)
      manual_params = update.apply_updates(manual_params, updates)
      states = new_states

    # Check equivalence.
    chex.assert_tree_all_close(manual_params, chain_params, rtol=1e-4)


def _map_keys_fn(fn):
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn


class MultiTransformTest(chex.TestCase):
  """Tests for the multi_transform wrapper."""

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_multi_transform(self, use_fn):
    params = {'a1': 1., 'b1': 2., 'z1': {'a2': 3., 'z2': {'c1': 4.}}}
    params = jax.tree_util.tree_map(jnp.asarray, params)
    input_updates = jax.tree_util.tree_map(lambda x: x / 10.0, params)
    tx_dict = {'a': transform.scale(-1.0),
               'b': transform.ema(0.0),  # stateful
               'c': transform.scale(2.0)}
    param_labels = _map_keys_fn(lambda k, _: k[0])
    if not use_fn:
      param_labels = param_labels(params)
    tx = combine.multi_transform(tx_dict, param_labels)
    update_fn = self.variant(tx.update)
    state = self.variant(tx.init)(params)

    correct_update_fn = _map_keys_fn(
        lambda k, v: {'a': -v, 'b': v, 'c': 2.0*v}[k[0]])

    updates, state = update_fn(input_updates, state, params)
    correct_updates = correct_update_fn(input_updates)
    chex.assert_tree_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = correct_update_fn(correct_updates)
    updates, state = update_fn(updates, state)
    chex.assert_tree_all_close(updates, correct_updates)

  @parameterized.parameters(list, tuple, dict)
  def test_empty(self, container):
    init_fn, update_fn = combine.multi_transform(
        {0: alias.sgd(1.)}, lambda _: 0)
    updates, _ = update_fn(container(), init_fn(container()))
    self.assertEqual(updates, container())

  @chex.all_variants
  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True))
  def test_labels_mismatch(self, use_extra_label, use_fn):
    # The labels from label_fn must be a subet of the keys for the tx.
    params = {'a': 1., 'b': [2., 3.], 'c': {'d': 4., 'e': (5., 6.)}}
    params = jax.tree_util.tree_map(jnp.asarray, params)
    label_tree = {'a': 0, 'b': [1, 0], 'c': 1}  # prefix of params

    if use_extra_label:
      label_tree['a'] = 3

    transforms = {0: alias.sgd(1.),
                  1: alias.adam(1., b1=0., b2=0.),
                  2: transform.trace(1.0)}
    init_fn, update_fn = combine.multi_transform(
        transforms, (lambda _: label_tree) if use_fn else label_tree)

    if use_extra_label:
      with self.assertRaises(ValueError):
        self.variant(init_fn)(params)
    else:
      state = self.variant(init_fn)(params)
      updates = jax.tree_util.tree_map(lambda x: x / 10.0, params)
      self.variant(update_fn)(updates, state)


if __name__ == '__main__':
  absltest.main()
