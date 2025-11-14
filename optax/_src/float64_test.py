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
"""Tests that types are preserved by the `update` calls when jax_enbable_x64."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from optax import transforms

from optax._src import alias
from optax._src import base
from optax._src import transform
from optax._src import update


ALL_MODULES = [
    ('identity', base.identity, {}),
    ('clip', transforms.clip, {'max_delta': 1.0}),
    ('clip_by_global_norm', transforms.clip_by_global_norm, {'max_norm': 1.0}),
    ('trace', transforms.trace, {'decay': 0.5, 'nesterov': False}),
    ('trace_with_nesterov', transforms.trace, {'decay': 0.5, 'nesterov': True}),
    ('scale_by_rss', transform.scale_by_rss, {}),
    ('scale_by_rms', transform.scale_by_rms, {}),
    ('scale_by_stddev', transform.scale_by_stddev, {}),
    ('scale_by_adam', transform.scale_by_adam, {}),
    ('scale', transform.scale, {'step_size': 3.0}),
    (
        'add_decayed_weights',
        transforms.add_decayed_weights,
        {'weight_decay': 0.1},
    ),
    (
        'scale_by_schedule',
        transform.scale_by_schedule,
        {'step_size_fn': lambda x: x * 0.1},
    ),
    ('scale_by_trust_ratio', transform.scale_by_trust_ratio, {}),
    ('add_noise', transforms.add_noise, {'eta': 1.0, 'gamma': 0.1, 'key': 0}),
    ('apply_every_k', transform.apply_every, {}),
    ('adagrad', alias.adagrad, {'learning_rate': 0.1}),
    ('adam', alias.adam, {'learning_rate': 0.1}),
    ('adamw', alias.adamw, {'learning_rate': 0.1}),
    ('fromage', alias.fromage, {'learning_rate': 0.1}),
    ('lamb', alias.lamb, {'learning_rate': 0.1}),
    ('noisy_sgd', alias.noisy_sgd, {'learning_rate': 0.1, 'key': 0}),
    ('rmsprop', alias.rmsprop, {'learning_rate': 0.1}),
    ('sgd', alias.sgd, {'learning_rate': 0.1}),
    ('sign_sgd', alias.sgd, {'learning_rate': 0.1}),
]


class Float64Test(parameterized.TestCase):

  def _assert_dtype_equals(self, tree1, tree2):
    tree1_types = jax.tree.map(lambda t: t.dtype, tree1)
    tree2_types = jax.tree.map(lambda t: t.dtype, tree2)
    self.assertEqual(tree1_types, tree2_types)

  @parameterized.named_parameters(ALL_MODULES)
  def test_mixed_dtype_input_outputs(self, transform_constr, transform_kwargs):
    jax.config.update('jax_enable_x64', True)
    initial_params = (
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jnp.array([3.0, 4.0], dtype=jnp.float64),
    )
    updates = (
        jnp.array([10.0, 21.0], dtype=jnp.float32),
        jnp.array([33.0, 42.0], dtype=jnp.float64),
    )
    scaler = transform_constr(**transform_kwargs)
    init_fn = jax.jit(scaler.init)
    update_fn = jax.jit(scaler.update)

    initial_state = init_fn(initial_params)
    updates, new_state = update_fn(
        updates, initial_state, params=initial_params
    )
    new_params = update.apply_updates(initial_params, updates)

    self._assert_dtype_equals(initial_state, new_state)
    self._assert_dtype_equals(initial_params, new_params)
    jax.config.update('jax_enable_x64', False)


if __name__ == '__main__':
  absltest.main()
