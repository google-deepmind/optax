# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the SAM optimizer in `sam.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import combine
from optax._src import numerics
from optax._src import test_utils
from optax._src import update
from optax.contrib import _sam
import optax.tree

_BASE_OPTIMIZERS_UNDER_TEST = [
    {'base_opt_name': 'sgd', 'base_opt_kwargs': {'learning_rate': 1e-3}},
]
_ADVERSARIAL_OPTIMIZERS_UNDER_TEST = [
    {'adv_opt_name': 'sgd', 'adv_opt_kwargs': {'learning_rate': 1e-5}},
    {'adv_opt_name': 'adam', 'adv_opt_kwargs': {'learning_rate': 1e-4}},
]


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


class SAMTest(parameterized.TestCase):

  @parameterized.product(
      _BASE_OPTIMIZERS_UNDER_TEST,
      _ADVERSARIAL_OPTIMIZERS_UNDER_TEST,
      sync_period=(2,),
      target=(_setup_parabola,),
      dtype=('float32',),
      opaque_mode=(False, True),
  )
  def test_optimization(
      self,
      base_opt_name,
      base_opt_kwargs,
      adv_opt_name,
      adv_opt_kwargs,
      sync_period,
      target,
      dtype,
      opaque_mode,
  ):
    dtype = jnp.dtype(dtype)
    base_opt = getattr(alias, base_opt_name)(**base_opt_kwargs)
    adv_opt = combine.chain(
        _sam.normalize(), getattr(alias, adv_opt_name)(**adv_opt_kwargs)
    )
    opt = _sam.sam(
        base_opt, adv_opt, sync_period=sync_period, opaque_mode=opaque_mode
    )
    initial_params, final_params, get_updates = target(dtype)

    if opaque_mode:
      update_kwargs = {'grad_fn': lambda p, _: get_updates(p)}
    else:
      update_kwargs = {}

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    # A no-op change, to verify that tree map works.
    state = optax.tree.map_params(opt, lambda v: v, state)

    for _ in range(25000 * sync_period):
      params, state = step(params, state)

    test_utils.assert_trees_all_close(
        params, final_params, rtol=3e-2, atol=3e-2)

  @parameterized.product(opaque_mode=(False, True))
  def test_sync_period_one_matches_base_optimizer(self, opaque_mode):
    # With sync_period=1 there is no adversarial step, so every update must be
    # exactly the update of the base optimizer from the current parameters.
    base_opt = alias.sgd(learning_rate=1e-1)
    adv_opt = combine.chain(_sam.normalize(), alias.sgd(learning_rate=1e-1))
    opt = _sam.sam(base_opt, adv_opt, sync_period=1, opaque_mode=opaque_mode)
    initial_params, _, get_updates = _setup_parabola(jnp.dtype('float32'))

    if opaque_mode:
      update_kwargs = {'grad_fn': lambda p, _: get_updates(p)}
    else:
      update_kwargs = {}

    sam_params = initial_params
    sam_state = opt.init(sam_params)
    base_params = initial_params
    base_state = base_opt.init(base_params)
    for _ in range(5):
      sam_updates, sam_state = opt.update(
          get_updates(sam_params), sam_state, sam_params, **update_kwargs
      )
      sam_params = update.apply_updates(sam_params, sam_updates)

      base_updates, base_state = base_opt.update(
          get_updates(base_params), base_state, base_params
      )
      base_params = update.apply_updates(base_params, base_updates)

      test_utils.assert_trees_all_close(
          sam_params, base_params, rtol=1e-6, atol=1e-6)

  @parameterized.product(sync_period=(1, 2, 3), reset_state=(True, False))
  def test_transparent_mode_matches_opaque_mode(self, sync_period, reset_state):
    # The outer optimizer depends on the parameters through the weight decay
    # and is stateful, so this checks that the transparent mode feeds it the
    # parameters of the last sync and only keeps its state on the last step.
    base_opt = alias.adamw(learning_rate=1e-1, weight_decay=1e-2)
    adv_opt = combine.chain(_sam.normalize(), alias.adam(learning_rate=1e-1))
    initial_params, _, get_updates = _setup_parabola(jnp.dtype('float32'))
    grad_fn = lambda p, _: get_updates(p)

    transparent_opt = _sam.sam(
        base_opt, adv_opt, sync_period=sync_period, reset_state=reset_state
    )
    opaque_opt = _sam.sam(
        base_opt,
        adv_opt,
        sync_period=sync_period,
        reset_state=reset_state,
        opaque_mode=True,
    )
    transparent_params = initial_params
    transparent_state = transparent_opt.init(transparent_params)
    opaque_params = initial_params
    opaque_state = opaque_opt.init(opaque_params)
    for _ in range(5):
      for _ in range(sync_period):
        updates, transparent_state = transparent_opt.update(
            get_updates(transparent_params),
            transparent_state,
            transparent_params,
        )
        transparent_params = update.apply_updates(transparent_params, updates)

      updates, opaque_state = opaque_opt.update(
          get_updates(opaque_params),
          opaque_state,
          opaque_params,
          grad_fn=grad_fn,
      )
      opaque_params = update.apply_updates(opaque_params, updates)

      test_utils.assert_trees_all_close(
          transparent_params, opaque_params, rtol=1e-6, atol=1e-6
      )


if __name__ == '__main__':
  absltest.main()
