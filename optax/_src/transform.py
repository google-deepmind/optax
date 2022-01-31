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
  Both are frequently found in the optimisation literature.

  Args:
    decay: the decay rate for the trace of past updates.
    nesterov: whether to use Nesterov momentum.
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return TraceState(
        trace=jax.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, params=None):
    del params
    f = lambda g, t: g + decay * t
    new_trace = jax.tree_multimap(f, updates, state.trace)
    updates = (
        jax.tree_multimap(f, updates, new_trace) if nesterov else new_trace)
    new_trace = utils.cast_tree(new_trace, accumulator_dtype)
    return updates, TraceState(trace=new_trace)

  return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def _update_moment_per_elem_norm(updates, moments, decay, order):
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

  return jax.tree_multimap(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  bias_correction = 1 - decay**count
  return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def _reject_complex(params):
  if any(jnp.iscomplexobj(x) for x in jax.tree_leaves(params)):
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
  Both are frequently found in the optimisation literature.

  Args:
    decay: the decay rate for the exponential moving average.
    debias: whether to debias the transformed gradient.
    accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaState(
        count=jnp.zeros([], jnp.int32),
        ema=jax.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, params=None):
    del params
    updates = new_ema = _update_moment(updates, state.ema, decay, order=1)
    count_inc = utils.safe_int32_increment(state.count)
    if debias:
      updates = _bias_correction(new_ema, decay, count_inc)
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
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    sum_of_squares = jax.tree_map(
        lambda t: jnp.full_like(t, initial_accumulator_value), params)
    return ScaleByRssState(sum_of_squares=sum_of_squares)

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jax.tree_multimap(
        lambda g, t: _abs_sq(g) + t, updates, state.sum_of_squares)
    inv_sqrt_g_square = jax.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares)
    updates = jax.tree_multimap(
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
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    initial_scale: initial value for second moment

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    nu = jax.tree_map(
        lambda n: jnp.full_like(n, initial_scale), params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
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
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    initial_scale: initial value for second moment

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(
        lambda n: jnp.full_like(n, initial_scale), params)  # second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, decay, 1)
    nu = _update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = utils.cast_tree(_bias_correction(mu, b1, count_inc), mu_dtype)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


ScaleState = base.EmptyState


def scale(
    step_size: float
) -> base.GradientTransformation:
  """Scale updates by some fixed scalar `step_size`.

  Args:
    step_size: a scalar corresponding to a fixed scaling factor for updates.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return ScaleState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(lambda g: step_size * g, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_param_block_norm(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
  """Scale updates for each param block by the norm of that block's parameters.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: minimum scaling factor.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_multimap(
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
    min_scale: minimum scaling factor.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_map(
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of variance of grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    s = jax.tree_map(jnp.zeros_like, params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    prediction_error = jax.tree_multimap(lambda g, m: g-m, updates, state.mu)
    nu = _update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jax.tree_map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of variance of grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    initial_accumulator_value: The starting value for accumulators.
      Only positive values are allowed.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    value_like = lambda p: jnp.full_like(p, initial_accumulator_value)
    mu = jax.tree_map(value_like, params)  # First moment
    nu = jax.tree_map(value_like, params)  # Second Central moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = jax.tree_multimap(
        lambda g, v: v - (1 - b2) * jnp.sign(v - _abs_sq(g)) * _abs_sq(g),
        updates, state.nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    threshold: Threshold for variance tractability

  Returns:
    An (init_fn, update_fn) tuple.
  """

  ro_inf = 2./(1 - b2) - 1
  def _radam_update(params):
    ro = params[0]
    mu_hat = params[1]
    nu_hat = params[2]
    r = jnp.sqrt((ro - 4)*(ro - 2)*ro_inf/((ro_inf - 4)*(ro_inf - 2)*ro))
    updates = jax.tree_multimap(
        lambda m, v: r*m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    b2t = b2**count_inc
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
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
    weight_decay: a scalar weight decay rate.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_multimap(
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
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jax.tree_map(
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
    min_norm: minimum norm for params and gradient norms; by default is zero.
    trust_coefficient: a multiplier for the trust ratio.
    eps: additive constant added to the denominator for numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
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

    updates = jax.tree_multimap(_scale_update, updates, params)
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
    eta: base variance of the gaussian noise added to the gradient.
    gamma: decay exponent for annealing of the variance.
    seed: seed for random number generation.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    num_vars = len(jax.tree_leaves(updates))
    treedef = jax.tree_structure(updates)
    count_inc = numerics.safe_int32_increment(state.count)
    variance = eta / count_inc**gamma
    standard_deviation = jnp.sqrt(variance)
    all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_multimap(
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
    k: emit non-zero gradients every k steps, otherwise accumulate them.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    grad_acc = jax.tree_map(jnp.zeros_like, params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state, params=None):
    del params
    c = state.count % k
    acc = c != 0
    grad_acc = jax.tree_multimap(
        lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = jax.tree_map(lambda ga: emit * ga, grad_acc)
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
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    del params
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(_subtract_mean, updates)
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def zeros_for_dim(p):
    return [jnp.zeros([s]) for s in p.shape]

  def init_fn(params):
    _reject_complex(params)
    mu = jax.tree_map(zeros_for_dim, params)
    nu = jax.tree_map(jnp.zeros_like, params)
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
    mu = jax.tree_multimap(
        lambda g, v:  # pylint:disable=g-long-lambda
        [jnp.reshape(v[i], _expanded_shape(g.shape, i)) for i in range(g.ndim)],
        updates, state.mu)
    accum = jax.tree_multimap(_new_accum, updates, mu)
    accum_inv_sqrt = jax.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), accum)
    up = jax.tree_multimap(lambda g, a: g*a, updates, accum_inv_sqrt)
    nu = _update_moment(up, state.nu, b1, 1)
    mu = jax.tree_map(
        lambda g: [_new_mu(g, i) for i in range(g.ndim)], accum)

    return nu, ScaleBySM3State(mu=mu, nu=nu)

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
