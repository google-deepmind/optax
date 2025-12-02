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
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _adding
import optax.tree


class HutchinsonState(NamedTuple):
  key: jax.Array


def hutchinson_estimator_diag_hessian(random_seed: Optional[jax.Array] = None):
  """Returns a GradientTransformationExtraArgs computing the Hessian diagonal.

  The Hessian diagonal is estimated using Hutchinson's estimator, which is
  unbiased but has high variance.

  Args:
    random_seed: key used to generate random vectors.

  Returns:
    GradientTransformationExtraArgs
  """

  def init_fn(params):
    del params
    key = random_seed if random_seed is not None else jax.random.PRNGKey(0)
    return HutchinsonState(key=key)

  def update_fn(updates, state, params=None, obj_fn=None, **extra_args):
    del extra_args  # complies with signature of GradientTransformationExtraArgs
                    # but ignores the extra_args
    if params is None:
      raise ValueError("params must be provided to hutchinson update function.")
    if obj_fn is None:
      raise ValueError("obj_fn must be provided to hutchinson update function.")
    del updates
    key, subkey = jax.random.split(state.key)
    random_signs = optax.tree.random_like(
        subkey,
        params,
        jax.random.rademacher,
        dtype=jnp.float32,
    )
    random_signs = optax.tree.cast(random_signs,
                                   optax.tree.dtype(params, "lowest"))
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
    b1: jax.typing.ArrayLike = 0.965,
    b2: jax.typing.ArrayLike = 0.99,
    eps: jax.typing.ArrayLike = 1e-8,
    gamma: jax.typing.ArrayLike = 0.01,
    clip_threshold: Optional[jax.typing.ArrayLike] = 1.0,
    update_interval: jax.typing.ArrayLike = 10,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
    verbose: bool = False,
    print_win_rate_every_n_steps: jax.typing.ArrayLike = 0,
) -> base.GradientTransformationExtraArgs:
  """Sophia optimizer.

  See :func:`optax.contrib.sophia` for more details.

  Args:
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the hessian diagonal estimates. Keep in mind
      effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default `b2` of
      0.99 is effectively 0.999 because default `update_interval` is every 10.
    eps: Small constant to avoid division by zero.
    gamma: Normalizing constant for the hessian diagonal.
    clip_threshold: Threshold for clipping updates.
    update_interval: Interval for updating the hessian diagonal.
    hessian_diagonal_fn: GradientTransformation that computes the diagonal of
      the Hessian. Default is Hutchinson's estimator (sophia-h). If using more
      than one device, be sure this function properly averages the hessian
      diagonal across devices.
    mu_dtype: dtype of the first moment estimates.
    verbose: If True, print win rate every n steps.
    print_win_rate_every_n_steps: Print sophia win rate every n steps for
      diagnostic purposes. Authors state this value should stay between 0.1 and
      0.5 during training. If win rate is too low, try increasing `gamma`. 0 to
      turn off.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  hessian_diagonal_fn = base.with_extra_args_support(hessian_diagonal_fn)

  def init_fn(params):
    return SophiaState(
        count=jnp.zeros([], jnp.int32),
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        nu=optax.tree.zeros_like(params),
        hessian_fn_state=hessian_diagonal_fn.init(params),
    )

  def update_fn(updates, state: SophiaState, params=None, **hess_fn_kwargs):
    if params is None:
      raise ValueError("params must be provided to sophia's update function.")
    count_inc = numerics.safe_int32_increment(state.count)

    grads = updates

    # Sophia update
    mu = optax.tree.update_moment(updates, state.mu, b1, 1)
    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    updates = jax.tree.map(
        lambda m, h: m / jnp.maximum(gamma * h, eps), mu_hat, state.nu
    )
    if clip_threshold is not None:
      sum_not_clipped = jax.tree.reduce(
          lambda x, y: x + y,
          jax.tree.map(lambda u: jnp.sum(jnp.abs(u) < clip_threshold), updates),
      )
      if verbose:
        win_rate = sum_not_clipped / optax.tree.size(updates)
        jax.lax.cond(
            count_inc % print_win_rate_every_n_steps == 0,
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
      nu = optax.tree.update_moment(hessian_diag, nu, b2, 1)

      return hess_fn_state, nu

    hessian_fn_state, nu = jax.lax.cond(
        jnp.equal(state.count % update_interval, 0),
        update_hessian_diag,
        lambda h, n: (h, n),
        state.hessian_fn_state,
        state.nu,
    )

    # Cast momentum back to mu_dtype
    mu = optax.tree.cast(mu, mu_dtype)

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
    b1: jax.typing.ArrayLike = 0.965,
    b2: jax.typing.ArrayLike = 0.99,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 1e-4,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    gamma: jax.typing.ArrayLike = 0.01,
    clip_threshold: Optional[jax.typing.ArrayLike] = 1.0,
    update_interval: jax.typing.ArrayLike = 10,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
    verbose: bool = False,
    print_win_rate_every_n_steps: jax.typing.ArrayLike = 0,
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

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the hessian diagonal estimates. Keep in mind
      effective `b2` is `1 - (1 - b2) / update_interval`, e.g. default `b2` of
      0.99 is effectively 0.999 because default `update_interval` is every 10.
    eps: Small constant to avoid division by zero.
    weight_decay: Rate at which to decay weights.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    gamma: Normalizing constant for the hessian diagonal.
    clip_threshold: Threshold for clipping updates.
    update_interval: Interval for updating the hessian diagonal.
    hessian_diagonal_fn: GradientTransformation that computes the diagonal of
      the Hessian. Default is Hutchinson's estimator (sophia-h). If using more
      than one device, be sure this function properly averages the hessian
      diagonal across devices.
    mu_dtype: dtype of the first moment estimates.
    verbose: If True, print win rate every n steps.
    print_win_rate_every_n_steps: Print sophia win rate every n steps for
      diagnostic purposes. Authors state this value should stay between 0.1 and
      0.5 during training. If win rate is too low, try increasing `gamma`. 0 to
      turn off.

  Returns:
    optax.GradientTransformationExtraArgs

  References:
    Liu et al., `Sophia: A Scalable Stochastic Second-order Optimizer for
    Language Model Pre-training <https://arxiv.org/abs/2305.14342>`_, 2023

    `Levanter <https://www.github.com/stanford-crfm/levanter>`_

  .. note::
    We use a rademacher vector to estimate the diagonal of the Hessian, contrary
    to the original implementation which uses a normal random vector.
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
          verbose=verbose,
          print_win_rate_every_n_steps=print_win_rate_every_n_steps,
      ),
      _adding.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate),
  ]
  return combine.chain(*tx)
