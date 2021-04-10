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

from functools import reduce
from typing import NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base, clipping, utils

# pylint:disable=no-value-for-parameter


class TraceState(base.OptState):
  """Holds an aggregation of past updates."""
  trace: base.Params


def trace(
    decay: float,
    nesterov: bool = False
) -> base.GradientTransformation:
  """Compute a trace of past updates.

  Args:
    decay: the decay rate for the tracing of past updates.
    nesterov: whether to use Nesterov momentum.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return TraceState(trace=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    f = lambda g, t: g + decay * t
    update_trace = jax.tree_multimap(f, updates, state.trace)
    updates = (
        jax.tree_multimap(f, updates, update_trace)
        if nesterov else update_trace)
    return updates, TraceState(trace=update_trace)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRssState(base.OptState):
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
        lambda g, t: jnp.square(g) + t, updates, state.sum_of_squares)
    inv_sqrt_g_square = jax.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares)
    updates = jax.tree_multimap(
        lambda scale, g: scale * g, inv_sqrt_g_square, updates)
    return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(base.OptState):
  """State for exponential root mean-squared (RMS)-normalized updates."""
  nu: base.Updates


def _update_moment(updates, moments, decay, order):
  return jax.tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


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
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(base.OptState):
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
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, m, n: g * jax.lax.rsqrt(n - jnp.square(m) + eps), updates, mu,
        nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(base.OptState):
  """State for the Adam algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


# TODO(b/183478923): remove legacy reference.
_safe_int32_increment = utils.safe_int32_increment


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  bias_correction = 1 - decay**count
  return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0
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

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count_inc = utils.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
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

  def init_fn(_):
    return ScaleState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(lambda g: step_size * g, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByBeliefState(base.OptState):
  """State for the rescaling by AdaBelief algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 0.,
    eps_root: float = 1e-16
) -> base.GradientTransformation:
  """Rescale updates according to the AdaBelief algorithm.

  References:
    [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of variance of grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.

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
    nu = _update_moment(prediction_error, state.nu, b2, 2)
    count_inc = utils.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
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
    signed_sq = jax.tree_multimap(
        lambda g, v: jnp.sign(v - g**2)*g**2, updates, state.nu)
    nu = _update_moment(signed_sq, state.nu, b2, 2)
    count_inc = utils.safe_int32_increment(state.count)
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
    nu = _update_moment(updates, state.nu, b2, 2)
    count_inc = utils.safe_int32_increment(state.count)
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
    weight_decay: float = 0.0
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: a scalar weight decay rate.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_multimap(
        lambda g, p: g + weight_decay * p, updates, params)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


# TODO(b/180608630): Remove deprecated references.
AdditiveWeightDecayState = AddDecayedWeightsState
additive_weight_decay = add_decayed_weights


class ScaleByScheduleState(base.OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


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

  def init_fn(_):
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jax.tree_map(
        lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates)
    return updates, ScaleByScheduleState(
        count=utils.safe_int32_increment(state.count))

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByFromageState(base.OptState):
  """Maintains count for step-size scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


class ScaleByTrustRatioState(NamedTuple):
  """The scale and decay trust ratio transformation is stateless."""


def scale_by_trust_ratio(
    min_norm: float = 0.0
) -> base.GradientTransformation:
  """Scale updates by trust ratio`.

  References:
    [You et. al 2020](https://arxiv.org/abs/1904.00962)

  Args:
    min_norm: minimum norm for params and gradient norms; by default is zero.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleByTrustRatioState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def _scale_update(update, param):

      # Clip norms to minimum value, by default no clipping.
      param_norm = utils.safe_norm(param, min_norm)
      update_norm = utils.safe_norm(update, min_norm)
      trust_ratio = param_norm / update_norm

      # If no minimum norm clipping is used
      # Set trust_ratio to 1 in case where parameters would never be updated.
      zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
      safe_trust_ratio = jnp.where(
          zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

      return update * safe_trust_ratio

    updates = jax.tree_multimap(_scale_update, updates, params)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class AddNoiseState(base.OptState):
  """State for adding gradient noise. Contains a count for annealing."""
  count: jnp.ndarray
  rng_key: jnp.ndarray


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

  def init_fn(_):
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    num_vars = len(jax.tree_leaves(updates))
    treedef = jax.tree_structure(updates)
    count_inc = utils.safe_int32_increment(state.count)
    variance = eta / count_inc**gamma
    all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_multimap(
        lambda g, n: g + variance.astype(g.dtype) * n,
        updates, noise)
    return updates, AddNoiseState(count=count_inc, rng_key=all_keys[0])

  return base.GradientTransformation(init_fn, update_fn)


class ApplyEvery(base.OptState):
  """Contains a counter and a gradient accumulator."""
  count: jnp.ndarray
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
    count_inc = utils.safe_int32_increment(state.count)
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

  def init_fn(_):
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(_subtract_mean, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleBySM3State(base.OptState):
  """State for the SM3 algorithm."""
  mu: base.Updates
  nu: base.Updates

def scale_by_sm3(b1: float = 0.9,
                 b2: float = 1.0,
                 eps: float = 1e-8) -> base.GradientTransformation:
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
      return coeffs[0]*g**2 + coeffs[1]*reduce(jnp.minimum, v)

  def _new_mu(g, i):
    if g.ndim < 2:
      return g
    else:
      return jnp.max(g, axis=other_axes(i, g.ndim))

  def other_axes(idx, ndim):
    return list(range(idx)) + list(range(idx+1, ndim))

  def update_fn(updates, state, params=None):
    del params
    mu = jax.tree_multimap(lambda g, v:
                          [jnp.reshape(v[i], _expanded_shape(g.shape, i))
                          for i in range(g.ndim)],
                          updates, state.mu)
    accum = jax.tree_multimap(lambda g, v: _new_accum(g, v),
                              updates, mu)
    accum_inv_sqrt = jax.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), accum)
    up = jax.tree_multimap(lambda g, a: g*a, updates, accum_inv_sqrt)
    nu = _update_moment(up, state.nu, b1, 1)
    mu = jax.tree_map(lambda g: [_new_mu(g, i)
                      for i in range(g.ndim)], accum)

    return nu, ScaleBySM3State(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


# TODO(b/183800387): remove legacy aliases.
# These legacy aliases are here for checkpoint compatibility
# To be removed once checkpoints have updated.

# clip = clipping.clip
ClipState = clipping.ClipState
# clip_by_global_norm = clipping.clip_by_global_norm
ClipByGlobalNormState = clipping.ClipByGlobalNormState
# global_norm = utils.global_norm
# identity = base.identity
IdentityState = base.EmptyState
# keep_params_nonnegative = constrain.keep_params_nonnegative
# NonNegativeParamsState = constrain.NonNegativeParamsState
# OptState = base.OptState
# Params = base.Params
# Updates = base.Updates
# zero_nans = constrain.zero_nans
# ZeroNansState = constrain.ZeroNansState
