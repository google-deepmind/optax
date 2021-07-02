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
"""Tests for optax._src.constrain."""

from absl.testing import absltest

import chex
import jax.numpy as jnp

from optax._src import combine
from optax._src import constrain
from optax._src import transform
from optax._src import update

STEPS = 50
LR = 1e-2


class ConstraintsTest(chex.TestCase):

  def test_keep_params_nonnegative(self):
    grads = (jnp.array([500., -500., 0.]),
             jnp.array([500., -500., 0.]),
             jnp.array([500., -500., 0.]))

    params = (jnp.array([-1., -1., -1.]),
              jnp.array([1., 1., 1.]),
              jnp.array([0., 0., 0.]))

    # vanilla sgd
    opt = combine.chain(
        transform.trace(decay=0, nesterov=False), transform.scale(-LR))
    opt_state = opt.init(params)

    updates, _ = opt.update(grads, opt_state, params)
    new_params = update.apply_updates(params, updates)

    chex.assert_tree_all_close(new_params, (jnp.array([-6., 4., -1.]),
                                            jnp.array([-4., 6., 1.]),
                                            jnp.array([-5., 5., 0.])))

    # sgd with keeping parameters non-negative
    opt = combine.chain(
        transform.trace(decay=0, nesterov=False), transform.scale(-LR),
        constrain.keep_params_nonnegative())
    opt_state = opt.init(params)

    updates, _ = opt.update(grads, opt_state, params)
    new_params = update.apply_updates(params, updates)

    chex.assert_tree_all_close(new_params, (jnp.array([0., 4., 0.]),
                                            jnp.array([0., 6., 1.]),
                                            jnp.array([0., 5., 0.])))

  @chex.all_variants
  def test_zero_nans(self):
    params = (jnp.zeros([3]), jnp.zeros([3]), jnp.zeros([3]))

    opt = constrain.zero_nans()
    opt_state = self.variant(opt.init)(params)
    update_fn = self.variant(opt.update)

    chex.assert_tree_all_close(opt_state,
                               constrain.ZeroNansState((jnp.array(False),) * 3))

    # Check an upate with nans
    grads_with_nans = (jnp.ones([3]),
                       jnp.array([1., float('nan'), float('nan')]),
                       jnp.array([float('nan'), 1., 1.]))
    updates, opt_state = update_fn(grads_with_nans, opt_state)
    chex.assert_tree_all_close(
        opt_state,
        constrain.ZeroNansState(
            (jnp.array(False), jnp.array(True), jnp.array(True))))
    chex.assert_tree_all_close(
        updates,
        (jnp.ones([3]), jnp.array([1., 0., 0.]), jnp.array([0., 1., 1.])))

    # Check an upate with nans and infs
    grads_with_nans_infs = (jnp.ones([3]),
                            jnp.array([1., float('nan'),
                                       float('nan')]),
                            jnp.array([float('inf'), 1., 1.]))
    updates, opt_state = update_fn(grads_with_nans_infs, opt_state)
    chex.assert_tree_all_close(
        opt_state,
        constrain.ZeroNansState(
            (jnp.array(False), jnp.array(True), jnp.array(False))))
    chex.assert_tree_all_close(updates, (jnp.ones([3]), jnp.array(
        [1., 0., 0.]), jnp.array([float('inf'), 1., 1.])))

    # Check an upate with only good values
    grads = (jnp.ones([3]), jnp.ones([3]), jnp.ones([3]))
    updates, opt_state = update_fn(grads, opt_state)
    chex.assert_tree_all_close(
        opt_state,
        constrain.ZeroNansState(
            (jnp.array(False), jnp.array(False), jnp.array(False))))
    chex.assert_tree_all_close(updates, grads)


if __name__ == '__main__':
  absltest.main()
