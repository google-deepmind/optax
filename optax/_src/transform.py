# Lint as: python3
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

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union
import chex
import jax
import jax.numpy as jnp
from optax._src import schedule


# pylint:disable=no-value-for-parameter
OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

# Function used to initialise the transformation's state.
TransformInitFn = Callable[
    [Params],
    Union[OptState, Sequence[OptState]]]
# Function used to apply a transformation.
TransformUpdateFn = Callable[
    [Updates, OptState, Optional[Params]],
    Tuple[Updates, OptState]]


class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: TransformInitFn
  update: TransformUpdateFn


NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of'
    ' parameters, but you are not passing `params` when calling `update`.')


class IdentityState(OptState):
  """The `identity` transformation is stateless."""


def identity() -> GradientTransformation:
  """Stateless identity transformation that leaves input gradients untouched.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return IdentityState()

  def update_fn(updates, state, params=None):
    del params
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class ClipState(OptState):
  """The `clip` transformation is stateless."""


def clip(max_delta) -> GradientTransformation:
  """Clip updates element-wise, to be between -max_delta and +max_delta.

  Args:
    max_delta: the maximum absolute value for each element in the update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(
        lambda g: jnp.clip(g, -max_delta, max_delta), updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


def global_norm(updates: Updates) -> Updates:
  return jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


class ClipByGlobalNormState(OptState):
  """The `clip_by_global_norm` transformation is stateless."""


def clip_by_global_norm(max_norm) -> GradientTransformation:
  """Clip updates using their global norm.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    max_norm: the maximum global norm for an update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipByGlobalNormState()

  def update_fn(updates, state, params=None):
    del params
    g_norm = global_norm(updates)
    # TODO(b/163995078): revert back to the following (faster) implementation
    # once analysed how it affects backprop through update (e.g. meta-gradients)
    # g_norm = jnp.maximum(max_norm, g_norm)
    # updates = jax.tree_map(lambda t: (t / g_norm) * max_norm, updates)
    trigger = g_norm < max_norm
    updates = jax.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


def unitwise_norm(x):
  """Computes norms of each output unit separately."""
  if len(jnp.squeeze(x).shape) <= 1:  # Scalars and vectors
    axis = None
    keepdims = False
  # Note that this assumes parameters with a shape of length 3 are multihead
  # linear parameters--if you wish to apply AGC to 1D convs, you may need
  # to modify this line.
  elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
    axis = 0
    keepdims = True
  elif len(x.shape) == 4:  # Conv kernels of shape HWIO
    axis = [0, 1, 2,]
    keepdims = True
  else:
    raise ValueError(f'Got a parameter with shape not in [1, 2, 3, 4]! {x}')
  return jnp.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_clip(g_norm, max_norm, grad):
  """Applies gradient clipping unit-wise."""
  trigger = g_norm < max_norm
  # This little max(., 1e-6) is distinct from the normal eps and just prevents
  # division by zero. It technically should be impossible to engage.
  clipped_grad = grad * (max_norm / jnp.maximum(g_norm, 1e-6))
  return jnp.where(trigger, grad, clipped_grad)


def adaptive_grad_clip(clipping, eps=1e-3) -> GradientTransformation:
  """Clip updates to be at most clipping * parameter_norm, unit-wise.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

  Args:
    clipping: Maximum allowed ratio of update norm to parameter norm.
    eps: epsilon term to prevent clipping of zero-initialized params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipByGlobalNormState()

  def update_fn(updates, state, params):
    g_norm = jax.tree_map(unitwise_norm, updates)
    p_norm = jax.tree_map(unitwise_norm, params)
    # Maximum allowable norm
    max_norm = jax.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
    # If grad norm > clipping * param_norm, rescale
    updates = jax.tree_multimap(unitwise_clip, g_norm, max_norm, updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class TraceState(OptState):
  """Holds an aggregation of past updates."""
  trace: Params


def trace(decay: float, nesterov: bool) -> GradientTransformation:
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

  return GradientTransformation(init_fn, update_fn)


class ScaleByRssState(OptState):
  """State holding the sum of gradient squares to date."""
  sum_of_squares: Updates


def scale_by_rss(initial_accumulator_value: float = 0.1, eps: float = 1e-7):
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

  return GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(OptState):
  """State for exponential root mean-squared (RMS)-normalized updates."""
  nu: Updates


def _update_moment(updates, moments, decay, order):
  return jax.tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def scale_by_rms(decay: float = 0.9, eps: float = 1e-8):
  """Rescale updates by the root of the exp. moving avg of the square.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    nu = jax.tree_map(jnp.zeros_like, params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(OptState):
  """State for centered exponential moving average of squares of updates."""
  mu: Updates
  nu: Updates


def scale_by_stddev(
    decay: float = 0.9, eps: float = 1e-8) -> GradientTransformation:
  """Rescale updates by the root of the centered exp. moving average of squares.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, decay, 1)
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, m, n: g * jax.lax.rsqrt(n - jnp.square(m) + eps), updates, mu,
        nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(OptState):
  """State for the Adam algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: Updates
  nu: Updates


def _safe_int32_increment(count):
  """Increments int32 counter by one.

  Normally `max_int + 1` would overflow to `min_int`. This functions ensures
  that when `max_int` is reached the counter stays at `max_int`.

  Args:
    count: a counter to be incremented.

  Returns:
    a counter incremented by 1, or max_int if the maximum precision is reached.
  """
  chex.assert_type(count, jnp.int32)
  max_int32_value = jnp.iinfo(jnp.int32).max
  one = jnp.array(1, dtype=jnp.int32)
  return jnp.where(count < max_int32_value, count + one, max_int32_value)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  bias_correction = 1 - decay**count
  return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def scale_by_adam(b1: float = 0.9,
                  b2: float = 0.999,
                  eps: float = 1e-8,
                  eps_root: float = 0.0) -> GradientTransformation:
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
    count_inc = _safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


class ScaleState(NamedTuple):
  """The scale transformation is stateless."""


def scale(step_size: float) -> GradientTransformation:
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

  return GradientTransformation(init_fn, update_fn)


class ScaleByBeliefState(OptState):
  """State for the rescaling by AdaBelief algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: Updates
  nu: Updates


def scale_by_belief(
    b1: float = 0.9, b2: float = 0.999,
    eps: float = 0., eps_root: float = 1e-16) -> GradientTransformation:
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
    count_inc = _safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


def scale_by_yogi(
    b1: float = 0.9, b2: float = 0.999,
    eps: float = 1e-3, eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6) -> GradientTransformation:
  """Rescale updates according to the Adam algorithm.

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
    count_inc = _safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


def scale_by_radam(b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   eps_root: float = 0.0,
                   threshold: float = 5.0) -> GradientTransformation:
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
    count_inc = _safe_int32_increment(state.count)
    b2t = b2**count_inc
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.lax.cond(
        ro >= threshold, _radam_update, lambda _: mu_hat,
        (ro, mu_hat, nu_hat))
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


class AddDecayedWeightsState(NamedTuple):
  """The decay transformation is stateless."""


def add_decayed_weights(weight_decay: float = 0.0) -> GradientTransformation:
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
      raise ValueError(NO_PARAMS_MSG)
    updates = jax.tree_multimap(
        lambda g, p: g + weight_decay * p, updates, params)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


# TODO(b/180608630): Remove deprecated references.
AdditiveWeightDecayState = AddDecayedWeightsState
additive_weight_decay = add_decayed_weights


class ScaleByScheduleState(OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


def scale_by_schedule(step_size_fn: schedule.Schedule):
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
        count=_safe_int32_increment(state.count))

  return GradientTransformation(init_fn, update_fn)


class ScaleByFromageState(OptState):
  """Maintains count for step-size scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


def _safe_norm(x, min_norm):
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

  The gradients of jnp.maximum(jnp.linalg.norm(x), min_norm) at 0.0 is NaN,
  because jax will evaluate both branches of the jnp.maximum.

  The version in this function will return the correct gradient of 0.0 in this
  situation.

  Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
  """
  norm = jnp.linalg.norm(x)
  x = jnp.where(norm < min_norm, jnp.ones_like(x), x)
  return jnp.where(norm < min_norm, min_norm, jnp.linalg.norm(x))


class ScaleByTrustRatioState(NamedTuple):
  """The scale and decay trust ratio transformation is stateless."""


def scale_by_trust_ratio(min_norm: float = 0.0) -> GradientTransformation:
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
      raise ValueError(NO_PARAMS_MSG)

    def _scale_update(update, param):

      # Clip norms to minimum value, by default no clipping.
      param_norm = _safe_norm(param, min_norm)
      update_norm = _safe_norm(update, min_norm)
      trust_ratio = param_norm / update_norm

      # If no minimum norm clipping is used
      # Set trust_ratio to 1 in case where parameters would never be updated.
      zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
      safe_trust_ratio = jnp.where(
          zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

      return update * safe_trust_ratio

    updates = jax.tree_multimap(_scale_update, updates, params)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class AddNoiseState(OptState):
  """State for adding gradient noise. Contains a count for annealing."""
  count: jnp.ndarray
  rng_key: jnp.ndarray


def add_noise(eta: float, gamma: float, seed: int) -> GradientTransformation:
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
    count_inc = _safe_int32_increment(state.count)
    variance = eta / count_inc**gamma
    all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_multimap(
        lambda g, n: g + variance.astype(g.dtype) * n,
        updates, noise)
    return updates, AddNoiseState(count=count_inc, rng_key=all_keys[0])

  return GradientTransformation(init_fn, update_fn)


class ApplyEvery(OptState):
  """Contains a counter and a gradient accumulator."""
  count: jnp.ndarray
  grad_acc: Updates


def apply_every(k: int = 1) -> GradientTransformation:
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
    count_inc = _safe_int32_increment(state.count)
    return updates, ApplyEvery(count=count_inc % k, grad_acc=grad_acc)

  return GradientTransformation(init_fn, update_fn)


def _subtract_mean(g):
  if len(g.shape) > 1:
    return g - g.mean(tuple(range(1, len(g.shape))), keepdims=True)
  else:
    return g


class CentralState(OptState):
  """The `centralize` transformation is stateless."""


def centralize() -> GradientTransformation:
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

  return GradientTransformation(init_fn, update_fn)


class NonNegativeParamsState(OptState):
  """The `keep_params_nonnegative` transformation is stateless."""


def keep_params_nonnegative() -> GradientTransformation:
  """Modifies the updates to keep parameters non-negative, i.e. >= 0.

  This transformation ensures that parameters after the update will be
  larger than or equal to zero.
  In a chain of transformations, this should be the last one.

  WARNING: the transformation expects input params to be non-negative.
  When params is negative the transformed update will move them to 0.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return NonNegativeParamsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(NO_PARAMS_MSG)

    updates = jax.tree_multimap(
        lambda p, u: jnp.where((p + u) < 0., -p, u), params, updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class ZeroNansState(OptState):
  """Contains a tree.

  The entry `found_nan` has the same tree structure as that of the parameters.
  Each leaf is a single boolean which contains True iff a NaN was detected in
  the corresponding parameter array at the last call to `update`.
  """
  found_nan: Any


def zero_nans() -> GradientTransformation:
  """A transformation which replaces NaNs with 0.

  Zeroing values in gradients is guaranteed to produce a direction of
  non-increasing loss.

  The state of the transformation has the same tree structure as that of the
  parameters. Each leaf is a single boolean which contains True iff a NaN was
  detected in the corresponding parameter array at the last call to `update`.
  This state is not used by the transformation internally, but lets users be
  aware when NaNs have been zeroed out.

  Returns:
    A `GradientTransformation`.
  """

  def init_fn(params):
    return ZeroNansState(
        jax.tree_map(lambda p: jnp.array(False, dtype=jnp.bool_), params))

  def update_fn(updates, opt_state, params=None):
    del params
    opt_state = ZeroNansState(
        jax.tree_map(lambda p: jnp.any(jnp.isnan(p)), updates))
    updates = jax.tree_map(
        lambda p: jnp.where(jnp.isnan(p), jnp.zeros_like(p), p), updates)
    return updates, opt_state

  return GradientTransformation(init=init_fn, update=update_fn)
