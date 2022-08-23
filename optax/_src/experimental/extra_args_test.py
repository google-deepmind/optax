# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for extra_kwargs."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import transform
from optax._src.experimental import extra_args as extra


def scale_by_loss():
  """Scale the gradient by the absolute value of the loss."""

  def init_fn(params, *, extra_args):
    del params, extra_args
    return base.EmptyState()

  def update_fn(updates, state, params, *, extra_args):
    del params
    updates = jax.tree_util.tree_map(
        lambda u: u / extra_args['loss'], updates)
    return updates, state

  return extra.GradientTransformationWithExtraArgs(init_fn, update_fn)


class ExtraArgsTest(absltest.TestCase):

  def test_named_chain(self):
    tx = extra.named_chain(
        ('scale', transform.scale(0.1)),
        ('scale_by_policy_loss', scale_by_loss()),
        ('scale_by_value_loss', scale_by_loss()),
    )

    params = {'a': jnp.ones((4,))}
    grads = params
    extra_args = {
        'scale_by_policy_loss': {'loss': 0.01},
        'scale_by_value_loss': {'loss': 10.0}}

    opt_state = tx.init(params, extra_args=extra_args)
    updates, opt_state = tx.update(
        grads, opt_state, params, extra_args=extra_args)
    chex.assert_tree_all_close(updates, {'a': jnp.ones((4,))})


if __name__ == '__main__':
  absltest.main()
