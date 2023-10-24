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
"""An implementation of the SAM Optimizer from https://arxiv.org/abs/2010.01412.

One way to describe what SAM does is that it does some number of steps (usually
1) of adversarial updates, followed by an outer gradient update.

What this means is that we have to do a bunch of steps:

    #adversarial step
    params = params + sam_rho * normalize(gradient)

    # outer update step
    params = cache - learning_rate * gradient
    cache = params

The SAM Optimizer here is written to wrap an inner adversarial optimizer which
will do the individual steps, and then with a defined cadence, does the outer
update steps.

To use the SAM optimzier then, create an adversarial optimizer, here SGD with
a normalized gradient and then wrap it with SAM itself.

    lr = 0.01
    rho = 0.1
    adv_opt = optax.chain(normalize(), optax.sgd(rho))
    opt = sam(lr, adv_opt, sync_period=2)

This is the simple drop-in SAM optimizer from the paper.
"""

from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import update
from optax._src import utils

# As a helper for SAM we need a gradient normalizing transformation.

NormalizeState = base.EmptyState


def normalize() -> base.GradientTransformation:
  """Normalizes the gradient.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return NormalizeState()

  def update_fn(updates, state, params=None):
    del params
    g_norm = utils.global_norm(updates)
    updates = jax.tree_map(lambda g: g / g_norm, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


@chex.dataclass
class SAMState:
  """State of `GradientTransformation` returned by `sam`.

  Attributes:
    steps_since_sync: Number of adversarial steps taken since the last outer
      update.
    opt_state: State of the outer optimizer.
    adv_state: State of the inner adversarial optimizer.
    cache: a place to store the last out updates.
  """

  steps_since_sync: jax.Array
  opt_state: base.OptState
  adv_state: base.OptState
  cache: base.Params


def sam(
    optimizer: base.GradientTransformation,
    adv_optimizer: base.GradientTransformation,
    sync_period: int = 2,
    reset_state: bool = True,
) -> base.GradientTransformation:
  """Implementation of SAM (Smoothness Aware Minimization).

  Performs steps with the inner adversarial optimizer and periodically
  updates an outer set of true parameters.  By default, resets
  the state of the adversarial optimizer after syncronization.  For example:

      opt = optax.sgd(lr)
      adv_opt = optax.chain(normalize(), optax.sgd(rho))
      sam_opt = sam(opt, adv_opt, sync_period=2)

  Would implement the simple drop-in SAM version from the paper which uses
  an inner adversarial optimizer of a normalized sgd for one step.

  Arguments:
    optimizer: the outer optimizer.
    adv_optimizer: the inner adversarial optimizer.
    sync_period: int, how often to run the outer optimizer, defaults to 2, or
      every other step.
    reset_state: bool, whether to reset the state of the inner optimizer after
      every sync period, defaults to True.

  Returns:
    sam_optimizer: an optax GradientTransformation implementation of SAM.
  """

  if sync_period < 1:
    raise ValueError("Synchronization period must be >= 1.")

  def init_fn(params: base.Params) -> SAMState:
    return SAMState(
        steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
        opt_state=optimizer.init(params),
        adv_state=adv_optimizer.init(params),
        cache=params,
    )

  def update_fn(
      updates: base.Updates, state: SAMState, params: base.Params
  ) -> Tuple[base.Updates, SAMState]:
    adv_updates, adv_state = adv_optimizer.update(
        updates, state.adv_state, params
    )
    opt_updates, opt_state = optimizer.update(updates, state.opt_state, params)
    sync_next = state.steps_since_sync == sync_period - 1
    updates, cache = _sam_update(  # pytype: disable=wrong-arg-types
        opt_updates, adv_updates, state.cache, sync_next, params
    )
    if reset_state:
      # Jittable way of resetting the fast optimizer state if parameters
      # will be synchronized after this update step.
      initial_state = adv_optimizer.init(params)
      adv_state = jax.tree_map(
          lambda current, init: (1 - sync_next) * current + sync_next * init,
          adv_state,
          initial_state,
      )

    opt_state = jax.tree_map(
        lambda current, prev: sync_next * current + (1 - sync_next) * prev,
        opt_state,
        state.opt_state,
    )

    steps_since_sync = (state.steps_since_sync + 1) % sync_period
    return updates, SAMState(
        steps_since_sync=steps_since_sync,
        adv_state=adv_state, opt_state=opt_state, cache=cache,
    )

  return base.GradientTransformation(init_fn, update_fn)


def _sam_update(
    updates: base.Updates,
    adv_updates: base.Updates,
    cache: base.Params,
    sync_next: jax.Array,
    params: SAMState,
) -> Tuple[base.Updates, base.Params]:
  """Returns the updates according to a sam step."""

  def update_or_sync(updates, adv_updates, cache, params):
    # if not sync_next:
    #   params = params - sam_rho * adv_updates
    #   cache = cache
    # if sync_next:
    #   params = cache - params + updates
    #   cache = params  (notice this has to be the updated params).
    return (1 - sync_next) * (-adv_updates) + sync_next * (
        cache - params + updates
    )

  param_updates = jax.tree_map(
      update_or_sync,
      updates,
      adv_updates,
      cache,
      params,
  )
  new_cache = jax.lax.cond(
      sync_next,
      lambda p, u, c: update.apply_updates(p, u),
      lambda p, u, c: c,
      params,
      param_updates,
      cache,
  )
  return param_updates, new_cache
