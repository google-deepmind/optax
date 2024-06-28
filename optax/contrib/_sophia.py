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
"""Sophia optimizer.

A contributed implementation of the Sophia optimizer from "Sophia: A Scalable
Stochastic Second-order Optimizer for Language Model Pre-training"
(https://arxiv.org/abs/2305.14342) by Hong Liu, Zhiyuan Li, David Hall,
Percy Liang, and Tengyu Ma.

This contribution is heavily based on the implementation of Sophia by levanter
(https://github.com/stanford-crfm/levanter) with some changes.
"""
from typing import Any, NamedTuple, Optional, Union, Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from optax._src import base
from optax._src import transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain
from optax import tree_utils as otu


class HutchinsonState(NamedTuple):
  key: PRNGKey


def hutchinson_estimator_diag_hessian(
    random_seed: Optional[PRNGKey] = None
):
  """Returns a GradientTransformation that computes the diagonal of the Hessian.

  The Hessian diagonal is estimated using Hutchinson's estimator, which is
  unbiased but has high variance.

  Args:
    random_seed: Optional[PRNGKey], key used to generate random vectors.

  Returns:
    GradientTransformationExtraArgs
  """

  def init_fn(params):
    del params
    key = random_seed if random_seed is not None else jax.random.PRNGKey(0)
    return HutchinsonState(key=key)

  def update_fn(updates, state, params=None, obj_fn=None):
    if params is None:
      raise ValueError("params must be provided to hutchinson update function.")
    if obj_fn is None:
      raise ValueError("obj_fn must be provided to hutchinson update function.")
    del updates
    key, subkey = jax.random.split(state.key)
    random_signs = otu.tree_random_like(
        subkey, params, partial(jax.random.rademacher, dtype=jnp.float32)
    )
    hvp = jax.jvp(jax.grad(obj_fn), (params,), (random_signs,))[1]
    product = jax.tree.map(lambda h, r: h * r, hvp, random_signs)
    return product, HutchinsonState(key=key)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)



class SophiaState(NamedTuple):
  """State for Sophia Optimizer."""

  count: jax.Array  # shape=(), dtype=jnp.int32
  mu: base.Updates  # momentum
  nu: base.Updates  # EMA of hessian diagonal
  hessian_fn_state: Any


def scale_by_sophia(
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
  """Sophia optimizer.

  A separate GradientTransformation is required through the argument
  `hessian_diagonal_fn` to compute the diagonal of the Hessian. Any extra
  arguments required by the hessian_diagonal_fn's update function can be
  passed through sophia's update function as trailing keyword arguments
  (**kwargs). The default hessian_diagonal_fn is Hutchinson's estimator
  and needs the objective function as an extra argument, `obj_fn`.
  obj_fn must accept `params` as its only argument and return only a
  scalar (the loss).

  For example, assuming your experiment's loss function is
  `loss_fn(params, batch) -> loss, aux` that takes multiple arguments and
  returns multiple outputs, we must modify it to `loss_fn(params) -> loss`:

  `obj_fn = lambda params: loss_fn(params, batch)[0]`

  where `batch` is the current step's batch.

  Then it can be passed to sophia's update function (which will pass it to the
  hessian_diagonal_fn's update function):

  `updates, state = sophia.update(updates, state, params, obj_fn=sophia_obj_fn)`

  Optionally, you can write your own GradientTransformation to compute the
  hessian diagonal. Use this file's hutchinson_estimator_diag_hessian function
  as an example. If you are using more than one device, be sure the hessian
  diagonal function properly averages the hessian diagonal across devices.
  The default hessian_diagonal_fn does not do this, and would cause params to
  diverge from each other across devices if using pmap for example.


  References:
    Liu et al., `Sophia: A Scalable Stochastic Second-order Optimizer for
    Language Model Pre-training <https://arxiv.org/abs/2305.14342>`_, 2023

    `Levanter <https://www.github.com/stanford-crfm/levanter>`_

  Args:
    b1: float, Exponential decay rate for the first moment estimates.
    b2: float, Exponential decay rate for the hessian diagonal estimates. Keep
        in mind effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default
        `b2` of 0.99 is effectively 0.999 because default `update_interval` is
        every 10.
    eps: float, Small constant to avoid division by zero.
    gamma: float, Normalizing constant for the hessian diagonal.
    clip_threshold: Optional[float], Threshold for clipping updates.
    update_interval: int, Interval for updating the hessian diagonal.
    hessian_diagonal_fn: GradientTransformation, GradientTransformation that
        computes the diagonal of the Hessian. Default is Hutchinson's estimator
        (sophia-h). If using more than one device, be sure this function
        properly averages the hessian diagonal across devices.
    mu_dtype: dtype of the first moment estimates.
    print_win_rate_every_n_steps: int, Print sophia win rate every n steps for
        diagnostic purposes. Authors state this value should stay between
        0.1 and 0.5 during training. If win rate is too low, try increasing
        `gamma`. 0 to turn off.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  mu_dtype = canonicalize_dtype(mu_dtype)

  def init_fn(params):
    return SophiaState(
        count=jnp.zeros([], jnp.int32),
        mu=otu.tree_zeros_like(params, dtype=mu_dtype),
        nu=otu.tree_zeros_like(params),
        hessian_fn_state=hessian_diagonal_fn.init(params),
    )

  def update_fn(
      updates, state: SophiaState, params=None, **hess_fn_kwargs
  ):
    if params is None:
      raise ValueError("params must be provided to sophia's update function.")
    count_inc = safe_int32_increment(state.count)

    grads = updates

    # Sophia update
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
          lambda: jax.debug.print("Sophia optimizer win rate: {}", win_rate),
          lambda: None,
      )

      updates = jax.tree.map(
          lambda u: jnp.clip(u, -clip_threshold, clip_threshold), updates
      )

    # Hessian diagonal update
    def update_hessian_diag(hess_fn_state, nu):
      hessian_diag, hess_fn_state = hessian_diagonal_fn.update(
          grads, hess_fn_state, params=params, **hess_fn_kwargs
      )

      # ema of hessian diagonal
      nu = otu.tree_update_moment(hessian_diag, nu, b2, 1)

      return hess_fn_state, nu

    hessian_fn_state, nu = jax.lax.cond(
        jnp.equal(state.count % update_interval, 0),
        update_hessian_diag,
        lambda h, n: (h, n),
        state.hessian_fn_state,
        state.nu,
    )

    # Cast momentum back to mu_dtype
    mu = otu.tree_cast(mu, mu_dtype)

    state = SophiaState(
        count=count_inc,
        mu=mu,
        nu=nu,
        hessian_fn_state=hessian_fn_state,
    )
    return updates, state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def sophia(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
  """Sophia optimizer.

  A separate GradientTransformation is required through the argument
  `hessian_diagonal_fn` to compute the diagonal of the Hessian. Any extra
  arguments required by the hessian_diagonal_fn's update function can be
  passed through sophia's update function as trailing keyword arguments
  (**kwargs). The default hessian_diagonal_fn is Hutchinson's estimator
  and needs the objective function as an extra argument, `obj_fn`.
  obj_fn must accept `params` as its only argument and return only a
  scalar (the loss).

  For example, assuming your experiment's loss function is
  `loss_fn(params, batch) -> loss, aux` that takes multiple arguments and
  returns multiple outputs, we must modify it to `loss_fn(params) -> loss`:

  `obj_fn = lambda params: loss_fn(params, batch)[0]`

  where `batch` is the current step's batch.

  Then it can be passed to sophia's update function (which will pass it to the
  hessian_diagonal_fn's update function):

  `updates, state = sophia.update(updates, state, params, obj_fn=sophia_obj_fn)`

  Optionally, you can write your own GradientTransformation to compute the
  hessian diagonal. Use this file's hutchinson_estimator_diag_hessian function
  as an example. If you are using more than one device, be sure the hessian
  diagonal function properly averages the hessian diagonal across devices.
  The default hessian_diagonal_fn does not do this, and would cause params to
  diverge from each other across devices if using pmap for example.


  References:
    Liu et al., `Sophia: A Scalable Stochastic Second-order Optimizer for
    Language Model Pre-training <https://arxiv.org/abs/2305.14342>`_, 2023

    `Levanter <https://www.github.com/stanford-crfm/levanter>`_

  Args:
    b1: float, Exponential decay rate for the first moment estimates.
    b2: float, Exponential decay rate for the hessian diagonal estimates. Keep
        in mind effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default
        `b2` of 0.99 is effectively 0.999 because default `update_interval` is
        every 10.
    eps: float, Small constant to avoid division by zero.
    gamma: float, Normalizing constant for the hessian diagonal.
    clip_threshold: Optional[float], Threshold for clipping updates.
    update_interval: int, Interval for updating the hessian diagonal.
    hessian_diagonal_fn: GradientTransformation, GradientTransformation that
        computes the diagonal of the Hessian. Default is Hutchinson's estimator
        (sophia-h). If using more than one device, be sure this function
        properly averages the hessian diagonal across devices.
    mu_dtype: dtype of the first moment estimates.
    print_win_rate_every_n_steps: int, Print sophia win rate every n steps for
        diagnostic purposes. Authors state this value should stay between
        0.1 and 0.5 during training. If win rate is too low, try increasing
        `gamma`. 0 to turn off.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  tx = [
      scale_by_sophia(
          b1=b1,
          b2=b2,
          eps=eps,
          gamma=gamma,
          clip_threshold=clip_threshold,
          update_interval=update_interval,
          hessian_diagonal_fn=hessian_diagonal_fn,
          mu_dtype=mu_dtype,
          print_win_rate_every_n_steps=print_win_rate_every_n_steps,
      ),
      transform.add_decayed_weights(weight_decay, mask=mask),
      transform.scale_by_learning_rate(learning_rate),
  ]
  return chain(*tx)
