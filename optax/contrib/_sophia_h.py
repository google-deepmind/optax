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
"""Sophia-H optimizer.

A contributed implementation of the Sophia-H optimizer from "Sophia: A Scalable
Stochastic Second-order Optimizer for Language Model Pre-training"
(https://arxiv.org/abs/2305.14342) by Hong Liu, Zhiyuan Li, David Hall,
Percy Liang, and Tengyu Ma.

This contribution is heavily based on the implementation of Sophia by levanter
(https://github.com/stanford-crfm/levanter) with some changes.
"""
from typing import Any, NamedTuple, Optional, Union, Callable

import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from optax._src import base
from optax._src import transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain
from optax import tree_utils as otu


class SophiaHState(NamedTuple):
  """State for Sophia-H and similar."""

  count: jax.Array  # shape=(), dtype=jnp.int32
  mu: Optional[base.Updates]  # momentum
  nu: base.Updates  # EMA of hessian
  key: PRNGKey


def scale_by_sophia_h(
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    mu_dtype: Optional[Any] = None,
    pmap_axis_name: Optional[str] = None,
    seed: PRNGKey = PRNGKey(0),
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
  """Sophia optimizer with hutchinson's estimator for the hessian diagonal.

  The loss function must be passed into sophia's update function to calculate
  the hessian diagonal. It must accept `params` as its only argument and return
  only a scalar (the loss).

  For example, considering loss function `loss_fn(params, batch) -> loss, aux`
  that takes multiple arguments and returns multiple outputs, we must modify it
  to `loss_fn(params) -> loss` like below:

  `sophia_obj_fn = lambda params: loss_fn(params, batch)[0]`

  Then it can be passed to sophia's update function:

  `updates, state = sophia.update(updates, state, params=params, obj_fn=sophia_obj_fn)`

  Notes:
    - TODO filter non-differentiable inputs to jvp
    - Paper uses gaussians for hutchinson's estimator but we use rademacher, see
      https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
      and https://x.com/dlwh/status/1785068009016672566
    - If using `jax.pmap`, providing the pmap axis name will perform separate
      monte carlo samples on each device for hutchinson's estimator for (almost)
      free, theoretically increasing hessian estimation accuracy.

  References:
    Liu et al., `Sophia: A Scalable Stochastic Second-order Optimizer for
    Language Model Pre-training <https://arxiv.org/abs/2305.14342>`_, 2023

    `Levanter <https://www.github.com/stanford-crfm/levanter>`_

  Args:
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the hessian diagonal estimates. Keep in mind
        effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default `b2`
        of 0.99 is effectively 0.999 because default `update_interval` is every
        10.
    eps: Small constant to avoid division by zero.
    gamma: Normalizing constant for the hessian diagonal.
    clip_threshold: Threshold for clipping updates.
    update_interval: Interval for updating the hessian diagonal.
    mu_dtype: dtype of the first moment estimates.
    pmap_axis_name: Provide pmap axis name if using pmap to perform separate
        monte carlo samples on each device for hutchinson's estimator for
        (almost) free.
    seed: PRNGKey.
    print_win_rate_every_n_steps: Print sophia win rate every n steps for
        diagnostic purposes. Authors state this value should stay between
        0.1 and 0.5 during training. If win rate is too low, try increasing
        `gamma`.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  mu_dtype = canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
    )
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)
    key = seed
    if pmap_axis_name and jax.local_device_count() > 1:
      key = jax.random.split(key, jax.local_device_count())
    return SophiaHState(
        count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, key=key
    )

  def update_fn(
      updates, state: SophiaHState, params=None, obj_fn=None
  ):
    if params is None:
      raise ValueError("params must be provided to sophia's update function.")
    if obj_fn is None:
      raise ValueError(
          "obj_fn must be provided to sophia's update function. "
          "See optimizer docstring for more information."
      )
    count_inc = safe_int32_increment(state.count)

    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    updates = jax.tree.map(
        lambda m, h: m / jnp.maximum(gamma * h, eps), mu_hat, state.nu
    )
    if clip_threshold is not None:
      sum_not_clipped = jax.tree.reduce(
          lambda x, y: x + y,
          jax.tree.map(
              lambda u: jnp.sum(jnp.abs(u) < clip_threshold), updates
          ),
      )
      total_tree_size = sum(x.size for x in jax.tree.leaves(updates))
      win_rate = sum_not_clipped / total_tree_size
      jax.lax.cond(
          jnp.logical_and(
              print_win_rate_every_n_steps > 0,
              count_inc % print_win_rate_every_n_steps == 0,
          ),
          lambda: jax.debug.print(
              "Sophia optimizer win rate: {}", win_rate
          ),
          lambda: None,
      )

      updates = jax.tree.map(
          lambda u: jnp.clip(u, -clip_threshold, clip_threshold), updates
      )

    key, nu = update_hessian(
        state.key, state.count, state.nu, params, obj_fn
    )

    mu = otu.tree_cast(mu, mu_dtype)
    state = SophiaHState(count=count_inc, mu=mu, nu=nu, key=key)
    return updates, state

  def update_hessian(key, count, nu, params, obj_fn):
    def _do_update(key):
      if pmap_axis_name is not None and jax.local_device_count() > 1:
        # get current replica's key
        idx = jax.lax.axis_index(pmap_axis_name)
        key = jax.lax.dynamic_index_in_dim(key, idx, keepdims=False)

      key, subkey = jax.random.split(key)
      hess = _stochastic_hessian_diagonal(subkey, obj_fn, params)

      if pmap_axis_name is not None and jax.local_device_count() > 1:
        # mean hessians across devices and gather keys
        hess = jax.lax.pmean(hess, axis_name=pmap_axis_name)
        key = jax.lax.all_gather(key, axis_name=pmap_axis_name)

      # ema of hessian diagonal
      new_nu = otu.tree_update_moment(hess, nu, b2, 1)

      return key, new_nu

    def _dont_update(key):
      return key, nu

    return jax.lax.cond(
      jnp.equal(count % update_interval, 0),
      _do_update,
      _dont_update,
      key
    )

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def sophia_h(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    mu_dtype: Optional[Any] = None,
    pmap_axis_name: Optional[str] = None,
    seed: PRNGKey = PRNGKey(0),
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
  """Sophia optimizer with hutchinson's estimator for the hessian diagonal.

  The loss function must be passed into sophia's update function to calculate
  the hessian diagonal. It must accept `params` as its only argument and return
  only a scalar (the loss).

  For example, considering loss function `loss_fn(params, batch) -> loss, aux`
  that takes multiple arguments and returns multiple outputs, we must modify it
  to `loss_fn(params) -> loss` like below:

  `sophia_obj_fn = lambda params: loss_fn(params, batch)[0]`

  Then it can be passed to sophia's update function:

  `updates, state = sophia.update(updates, state, params=params, obj_fn=sophia_obj_fn)`

  Notes:
    - TODO filter non-differentiable inputs to jvp
    - Paper uses gaussians for hutchinson's estimator but we use rademacher, see
      https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
      and https://x.com/dlwh/status/1785068009016672566
    - If using `jax.pmap`, providing the pmap axis name will perform separate
      monte carlo samples on each device for hutchinson's estimator for (almost)
      free, theoretically increasing hessian estimation accuracy.

  References:
    Liu et al., `Sophia: A Scalable Stochastic Second-order Optimizer for
    Language Model Pre-training <https://arxiv.org/abs/2305.14342>`_, 2023

    `Levanter <https://www.github.com/stanford-crfm/levanter>`_

  Args:
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the hessian diagonal estimates. Keep in mind
        effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default `b2`
        of 0.99 is effectively 0.999 because default `update_interval` is every
        10.
    eps: Small constant to avoid division by zero.
    gamma: Normalizing constant for the hessian diagonal.
    clip_threshold: Threshold for clipping updates.
    update_interval: Interval for updating the hessian diagonal.
    mu_dtype: dtype of the first moment estimates.
    pmap_axis_name: Provide pmap axis name if using pmap to perform separate
        monte carlo samples on each device for hutchinson's estimator for
        (almost) free.
    seed: PRNGKey.
    print_win_rate_every_n_steps: Print sophia win rate every n steps for
        diagnostic purposes. Authors state this value should stay between
        0.1 and 0.5 during training. If win rate is too low, try increasing
        `gamma`.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  tx = [
    scale_by_sophia_h(
        b1=b1,
        b2=b2,
        eps=eps,
        gamma=gamma,
        clip_threshold=clip_threshold,
        update_interval=update_interval,
        mu_dtype=mu_dtype,
        pmap_axis_name=pmap_axis_name,
        seed=seed,
        print_win_rate_every_n_steps=print_win_rate_every_n_steps,
    ),
    transform.add_decayed_weights(weight_decay, mask=mask),
    transform.scale_by_learning_rate(learning_rate),
  ]
  return chain(*tx)


def _tree_rademacher_like(key, tree):
  leaves, structure = jax.tree.flatten(tree)
  keys = jax.random.split(key, len(leaves))
  g = jax.tree.map(
      lambda key, x: jax.random.rademacher(key, x.shape, dtype=jnp.float32),
      list(keys),
      leaves,
  )
  g = jax.tree.unflatten(structure, g)
  return g


def _stochastic_hessian_diagonal(key, obj_fn, model):
  gaussians = _tree_rademacher_like(key, model)
  product = jax.jvp(jax.grad(obj_fn), (model,), (gaussians,))[1]
  return jax.tree.map(
      lambda grad, gaussian: grad * gaussian, product, gaussians
  )
