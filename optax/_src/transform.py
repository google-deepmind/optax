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
"""Gradient transformations."""

import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import clipping
from optax._src import numerics
from optax._src import utils
from optax._src import wrappers

# pylint:disable=no-value-for-parameter

_abs_sq = numerics.abs_sq


class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: base.Params


def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Compute a trace of past updates.

  Note: `trace` and `ema` have very similar but distinct updates;
  `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
  Both are frequently found in the optimization literature.

  Args:
    decay: Decay rate for the trace of past updates.
    nesterov: Whether to use Nesterov momentum.
    accumulator_dtype: Optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return TraceState(
        trace=jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, params=None):
    del params
    f = lambda g, t: g + decay * t
    new_trace = jax.tree_util.tree_map(f, updates, state.trace)
    updates = (
        jax.tree_util.tree_map(f, updates, new_trace) if nesterov
        else new_trace)
    new_trace = utils.cast_tree(new_trace, accumulator_dtype)
    return updates, TraceState(trace=new_trace)

  return base.GradientTransformation(init_fn, update_fn)


def update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def update_infinity_moment(updates, moments, decay, eps):
  """Compute the exponential moving average of the infinity norm."""
  return jax.tree_util.tree_map(
      lambda g, t: jnp.maximum(jnp.abs(g) + eps, decay * t), updates, moments)


def update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return _abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)


def _reject_complex(params):
  if any(jnp.iscomplexobj(x) for x in jax.tree_util.tree_leaves(params)):
    raise ValueError('This transformation does not support complex parameters.')


class EmaState(NamedTuple):
  """Holds an exponential moving average of past updates."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  ema: base.Params


def ema(
    decay: float,
    debias: bool = True,
    accumulator_dtype: Optional[Any] = None
) -> base.GradientTransformation:
  """Compute an exponential moving average of past updates.

  Note: `trace` and `ema` have very similar but distinct updates;
  `ema = decay * ema + (1-decay) * t`, while `trace = decay * trace + t`.
  Both are frequently found in the optimization literature.

  Args:
    decay: Decay rate for the exponential moving average.
    debias: Whether to debias the transformed gradient.
    accumulator_dtype: Optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaState(
        count=jnp.zeros([], jnp.int32),
        ema=jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, params=None):
    del params
    updates = new_ema = update_moment(updates, state.ema, decay, order=1)
    count_inc = utils.safe_int32_increment(state.count)
    if debias:
      updates = bias_correction(new_ema, decay, count_inc)
    state_ema = utils.cast_tree(new_ema, accumulator_dtype)
    return updates, EmaState(count=count_inc, ema=state_ema)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRssState(NamedTuple):
  """State holding the sum of gradient squares to date."""
  sum_of_squares: base.Updates


def scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7
) -> base.GradientTransformation:
  """Rescale updates by the root of the sum of all squared gradients to date.

  References:
    [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    [McMahan et al., 2010](https://arxiv.org/abs/1002.4908)

  Args:
    initial_accumulator_value: Starting value for accumulators, must be >= 0.
    eps: A small floating point value to avoid zero denominator.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    sum_of_squares = jax.tree_util.tree_map(
        lambda t: jnp.full_like(t, initial_accumulator_value), params)
    return ScaleByRssState(sum_of_squares=sum_of_squares)

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jax.tree_util.tree_map(
        lambda g, t: _abs_sq(g) + t, updates, state.sum_of_squares)
    inv_sqrt_g_square = jax.tree_util.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares)
    updates = jax.tree_util.tree_map(
        lambda scale, g: scale * g, inv_sqrt_g_square, updates)
    return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(NamedTuple):
  """State for exponential root mean-squared (RMS)-normalized updates."""
  nu: base.Updates


def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.
) -> base.GradientTransformation:
  """Rescale updates by the root of the exp. moving avg of the square.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    nu = jax.tree_util.tree_map(
        lambda n: jnp.full_like(n, initial_scale), params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jax.tree_util.tree_map(
        lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(NamedTuple):
  """State for centered exponential moving average of squares of updates."""
  mu: base.Updates
  nu: base.Updates


def scale_by_stddev(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.
) -> base.GradientTransformation:
  """Rescale updates by the root of the centered exp. moving average of squares.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(
        lambda n: jnp.full_like(n, initial_scale), params)  # second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, decay, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jax.tree_util.tree_map(
        lambda g, m, n: g * jax.lax.rsqrt(n - _abs_sq(m) + eps),
        updates, mu, nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAmsgradState(NamedTuple):
  """State for the AMSGrad algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates
  nu_max: base.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  References:
    [Reddi et al, 2018](https://openreview.net/forum?id=ryQu7f-RZ)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    nu_max = jax.tree_util.tree_map(jnp.zeros_like, params)
    return ScaleByAmsgradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu,
                               nu_max=nu_max)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    nu_max = jax.tree_util.tree_map(jnp.maximum, state.nu_max, nu_hat)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_max)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAmsgradState(count=count_inc, mu=mu, nu=nu,
                                        nu_max=nu_max)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_adamax(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8
) -> base.GradientTransformation:
  """Rescale updates according to the Adamax algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted maximum of grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Infinite moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_int32_increment(state.count)
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_infinity_moment(updates, state.nu, b2, eps)
    # Bias correction for mean. No bias correction needed for infinity moment.
    mu_hat = bias_correction(mu, b1, count_inc)
    updates = jax.tree_util.tree_map(lambda m, v: m / v, mu_hat, nu)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


ScaleState = base.EmptyState


def scale(
    step_size: float
) -> base.GradientTransformation:
  """Scale updates by some fixed scalar `step_size`.

  Args:
    step_size: A scalar corresponding to a fixed scaling factor for updates.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return ScaleState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_util.tree_map(lambda g: step_size * g, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_param_block_norm(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
  """Scale updates for each param block by the norm of that block's parameters.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda u, p: u * numerics.safe_norm(p, min_scale),
        updates, params)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_param_block_rms(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
  """Scale updates by rms of the gradient for each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda u, p: u * numerics.safe_root_mean_squares(p, min_scale),
        updates, params)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByBeliefState(NamedTuple):
  """State for the rescaling by AdaBelief algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16
) -> base.GradientTransformation:
  """Rescale updates according to the AdaBelief algorithm.

  References:
    [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    s = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    prediction_error = jax.tree_util.tree_map(
        lambda g, m: g-m, updates, state.mu)
    nu = update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jax.tree_util.tree_map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_yogi(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6
) -> base.GradientTransformation:
  """Rescale updates according to the Yogi algorithm.

  Supports complex numbers, see
  https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29

  References:
    [Zaheer et al, 2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html) #pylint:disable=line-too-long

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    initial_accumulator_value: The starting value for accumulators.
      Only positive values are allowed.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    value_like = lambda p: jnp.full_like(p, initial_accumulator_value)
    mu = jax.tree_util.tree_map(value_like, params)  # First moment
    nu = jax.tree_util.tree_map(value_like, params)  # Second Central moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = jax.tree_util.tree_map(
        lambda g, v: v - (1 - b2) * jnp.sign(v - _abs_sq(g)) * _abs_sq(g),
        updates, state.nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0
) -> base.GradientTransformation:
  """Rescale updates according to the Rectified Adam algorithm.

  References:
    [Liu et al, 2020](https://arxiv.org/abs/1908.03265)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    threshold: Threshold for variance tractability.

  Returns:
    A `GradientTransformation` object.
  """

  ro_inf = 2./(1 - b2) - 1
  def _radam_update(params):
    ro = params[0]
    mu_hat = params[1]
    nu_hat = params[2]
    r = jnp.sqrt((ro - 4)*(ro - 2)*ro_inf/((ro_inf - 4)*(ro_inf - 2)*ro))
    updates = jax.tree_util.tree_map(
        lambda m, v: r*m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    b2t = b2**count_inc
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.lax.cond(
        ro >= threshold, _radam_update, lambda _: mu_hat,
        (ro, mu_hat, nu_hat))
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


AddDecayedWeightsState = base.EmptyState


def add_decayed_weights(
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda g, p: g + weight_decay * p, updates, params)
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask)
  return base.GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(NamedTuple):
  """Maintains count for scale scheduling."""
  count: chex.Array  # shape=(), dtype=jnp.int32


def scale_by_schedule(
    step_size_fn: base.Schedule
) -> base.GradientTransformation:
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: A function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jax.tree_util.tree_map(
        lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates)
    return updates, ScaleByScheduleState(
        count=numerics.safe_int32_increment(state.count))

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByFromageState(NamedTuple):
  """Maintains count for step-size scheduling."""
  count: chex.Array  # shape=(), dtype=jnp.int32


class ScaleByTrustRatioState(NamedTuple):
  """The scale and decay trust ratio transformation is stateless."""


def scale_by_trust_ratio(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.,
    eps: float = 0.,
) -> base.GradientTransformation:
  """Scale updates by trust ratio`.

  References:
    [You et. al 2020](https://arxiv.org/abs/1904.00962)

  Args:
    min_norm: Minimum norm for params and gradient norms; by default is zero.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Additive constant added to the denominator for numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return ScaleByTrustRatioState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def _scale_update(update, param):

      # Clip norms to minimum value, by default no clipping.
      param_norm = numerics.safe_norm(param, min_norm)
      update_norm = numerics.safe_norm(update, min_norm)
      trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

      # If no minimum norm clipping is used
      # Set trust_ratio to 1 in case where parameters would never be updated.
      zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
      safe_trust_ratio = jnp.where(
          zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

      return update * safe_trust_ratio

    updates = jax.tree_util.tree_map(_scale_update, updates, params)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class AddNoiseState(NamedTuple):
  """State for adding gradient noise. Contains a count for annealing."""
  count: chex.Array
  rng_key: chex.PRNGKey


def add_noise(
    eta: float,
    gamma: float,
    seed: int
) -> base.GradientTransformation:
  """Add gradient noise.

  References:
    [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

  Args:
    eta: Base variance of the gaussian noise added to the gradient.
    gamma: Decay exponent for annealing of the variance.
    seed: Seed for random number generation.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    num_vars = len(jax.tree_util.tree_leaves(updates))
    treedef = jax.tree_util.tree_structure(updates)
    count_inc = numerics.safe_int32_increment(state.count)
    variance = eta / count_inc**gamma
    standard_deviation = jnp.sqrt(variance)
    all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_util.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_util.tree_map(
        lambda g, n: g + standard_deviation.astype(g.dtype) * n,
        updates, noise)
    return updates, AddNoiseState(count=count_inc, rng_key=all_keys[0])

  return base.GradientTransformation(init_fn, update_fn)


class ApplyEvery(NamedTuple):
  """Contains a counter and a gradient accumulator."""
  count: chex.Array
  grad_acc: base.Updates


def apply_every(
    k: int = 1
) -> base.GradientTransformation:
  """Accumulate gradients and apply them every k steps.

  Note that if this transformation is part of a chain, the states of the other
  transformations will still be updated at every step. In particular, using
  `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
  to not using `apply_every` with a batch size of N. If this equivalence is
  important for you, consider using the `optax.MultiSteps`.

  Args:
    k: Emit non-zero gradients every k steps, otherwise accumulate them.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    grad_acc = jax.tree_util.tree_map(jnp.zeros_like, params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state, params=None):
    del params
    c = state.count % k
    acc = c != 0
    grad_acc = jax.tree_util.tree_map(
        lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = jax.tree_util.tree_map(lambda ga: emit * ga, grad_acc)
    count_inc = numerics.safe_int32_increment(state.count)
    return updates, ApplyEvery(count=count_inc % k, grad_acc=grad_acc)

  return base.GradientTransformation(init_fn, update_fn)


def _subtract_mean(g):
  if len(g.shape) > 1:
    return g - g.mean(tuple(range(1, len(g.shape))), keepdims=True)
  else:
    return g


CentralState = base.EmptyState


def centralize() -> base.GradientTransformation:
  """Centralize gradients.

  References:
    [Yong et al, 2020](https://arxiv.org/abs/2004.01461)

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_util.tree_map(_subtract_mean, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleBySM3State(NamedTuple):
  """State for the SM3 algorithm."""
  mu: base.Updates
  nu: base.Updates


def scale_by_sm3(
    b1: float = 0.9,
    b2: float = 1.0,
    eps: float = 1e-8
) -> base.GradientTransformation:
  """Scale updates by sm3`.

  References:
    [Anil et. al 2019](https://arxiv.org/abs/1901.11150)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def zeros_for_dim(p):
    return [jnp.zeros([s]) for s in p.shape]

  def init_fn(params):
    _reject_complex(params)
    mu = jax.tree_util.tree_map(zeros_for_dim, params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)
    return ScaleBySM3State(mu, nu)

  def _expanded_shape(shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _new_accum(g, v):
    coeffs = ((1.0 - b2) if b2 != 1.0 else 1.0, b2)
    if g.ndim < 2:
      return coeffs[0]*g**2 + coeffs[1]*v[0]
    else:
      return coeffs[0]*g**2 + coeffs[1]*functools.reduce(jnp.minimum, v)

  def _new_mu(g, i):
    if g.ndim < 2:
      return g
    else:
      return jnp.max(g, axis=other_axes(i, g.ndim))

  def other_axes(idx, ndim):
    return list(range(idx)) + list(range(idx+1, ndim))

  def update_fn(updates, state, params=None):
    del params
    mu = jax.tree_util.tree_map(
        lambda g, v:  # pylint:disable=g-long-lambda
        [jnp.reshape(v[i], _expanded_shape(g.shape, i)) for i in range(g.ndim)],
        updates, state.mu)
    accum = jax.tree_util.tree_map(_new_accum, updates, mu)
    accum_inv_sqrt = jax.tree_util.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), accum)
    up = jax.tree_util.tree_map(lambda g, a: g*a, updates, accum_inv_sqrt)
    nu = update_moment(up, state.nu, b1, 1)
    mu = jax.tree_util.tree_map(
        lambda g: [_new_mu(g, i) for i in range(g.ndim)], accum)

    return nu, ScaleBySM3State(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByNovogradState(NamedTuple):
  """State for Novograd."""
  count: chex.Array
  mu: base.Updates
  nu: base.Updates


def scale_by_novograd(
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Computes NovoGrad updates.

  References:
    [Ginsburg et al, 2019](https://arxiv.org/abs/1905.11286)

  Args:
    b1: A decay rate for the exponentially weighted average of grads.
    b2: A decay rate for the exponentially weighted average of squared grads.
    eps: A term added to the denominator to improve numerical stability.
    eps_root: A term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    weight_decay: A scalar weight decay rate.
    mu_dtype: An optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(lambda _: 0.0, params)  # Second moment
    return ScaleByNovogradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def nu_addition(grads):
    return jnp.linalg.norm(grads)**2

  def mu_addition(grads, params, nu):
    return grads / (jnp.sqrt(nu + eps_root) + eps) + weight_decay * params

  def init_nu(grads, nu):
    del nu
    return jax.tree_util.tree_map(nu_addition, grads)

  def update_nu(grads, nu):
    updates = jax.tree_util.tree_map(nu_addition, grads)
    return update_moment(updates, nu, b2, 1)

  def init_mu(grads, params, mu, nu):
    del mu
    return jax.tree_util.tree_map(mu_addition, grads, params, nu)

  def update_mu(grads, params, mu, nu):
    updates = jax.tree_util.tree_map(mu_addition, grads, params, nu)
    return jax.tree_util.tree_map(lambda m, u: b1 * m + u, mu, updates)

  # Second moment
  def update_fn(updates, state, params):
    count_inc = numerics.safe_int32_increment(state.count)

    nu = jax.lax.cond(count_inc == 1, init_nu, update_nu, updates, state.nu)

    mu = jax.lax.cond(count_inc == 1, init_mu, update_mu, updates, params,
                      state.mu, nu)

    mu = utils.cast_tree(mu, mu_dtype)
    updates = mu
    return updates, ScaleByNovogradState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_optimistic_gradient(alpha: float = 1.0,
                                 beta: float = 1.0
                                ) -> base.GradientTransformation:
  """Compute generalized optimistic gradients.

  References:
    [Mokhtari et al, 2019](https://arxiv.org/abs/1901.08511v2)

  Args:
    alpha: Coefficient for generalized optimistic gradient descent.
    beta: Coefficient for negative momentum.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    prev_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    return TraceState(trace=prev_grads)

  def update_fn(updates, state, params=None):
    del params

    new_updates = jax.tree_util.tree_map(
        lambda grad_t, grad_tm1: (alpha + beta) * grad_t - beta * grad_tm1,
        updates, state.trace)
    return new_updates, TraceState(trace=updates)

  return base.GradientTransformation(init_fn, update_fn)


# TODO(b/183800387): remove legacy aliases.
# These legacy aliases are here for checkpoint compatibility
# To be removed once checkpoints have updated.
_safe_int32_increment = numerics.safe_int32_increment
safe_int32_increment = numerics.safe_int32_increment
AdditiveWeightDecayState = AddDecayedWeightsState
additive_weight_decay = add_decayed_weights
ClipState = clipping.ClipState
ClipByGlobalNormState = clipping.ClipByGlobalNormState
