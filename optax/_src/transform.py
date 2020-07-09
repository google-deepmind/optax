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
import jax
import jax.numpy as jnp


# pylint:disable=no-value-for-parameter
OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.


class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: Callable[  # Function used to initialise the transformation's state.
      [Params], Union[OptState, Sequence[OptState]]]
  update: Callable[  # Function used to apply a transformation.
      [Updates, OptState, Optional[Params]], Tuple[Updates, OptState]]


InitUpdate = GradientTransformation


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
    updates = jax.tree_multimap(
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
    g_norm = jnp.maximum(max_norm, g_norm)
    updates = jax.tree_multimap(lambda t: (t / g_norm) * max_norm, updates)
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
    return TraceState(trace=jax.tree_multimap(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    f = lambda g, t: g + decay * t
    update_trace = jax.tree_multimap(f, updates, state.trace)
    updates = (
        jax.tree_multimap(f, updates, update_trace)
        if nesterov else update_trace)
    return updates, TraceState(trace=update_trace)

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
    nu = jax.tree_multimap(jnp.zeros_like, params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, n: g / (jnp.sqrt(n + eps)), updates, nu)
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
    mu = jax.tree_multimap(jnp.zeros_like, params)  # First moment
    nu = jax.tree_multimap(jnp.zeros_like, params)  # Second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, decay, 1)
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = jax.tree_multimap(
        lambda g, m, n: g / jnp.sqrt(n - jnp.square(m) + eps), updates, mu, nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(OptState):
  """State for the Adam algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: Updates
  nu: Updates


def scale_by_adam(
    b1: float = 0.9, b2: float = 0.999,
    eps: float = 1e-8, eps_root: float = 0.0) -> GradientTransformation:
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
    mu = jax.tree_multimap(jnp.zeros_like, params)  # First moment
    nu = jax.tree_multimap(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    mu_hat = jax.tree_multimap(lambda t: t / (1 - b1 ** (state.count + 1)), mu)
    nu_hat = jax.tree_multimap(lambda t: t / (1 - b2 ** (state.count + 1)), nu)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=state.count + 1, mu=mu, nu=nu)

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
    updates = jax.tree_multimap(lambda g: step_size * g, updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class ScaleAndDecayState(NamedTuple):
  """The scale and decay transformation is stateless."""


def scale_and_decay(step_size: float,
                    weight_decay: float = 0.0) -> GradientTransformation:
  """Scale updates by `step_size`, and decay parameters by `weight_decay`.

  Args:
    step_size: a scalar corresponding to a fixed scaling factor for updates.
    weight_decay: a scalar weight decay rate, which is further scaled by the
      step size.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleAndDecayState()

  def update_fn(updates, state, params):
    updates = jax.tree_multimap(
        lambda g, p: (step_size * g - abs(step_size) * weight_decay * p),
        updates, params)
    return updates, state

  return GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


def scale_by_schedule(step_size_fn: Callable[[jnp.ndarray], jnp.ndarray]):
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
    updates = jax.tree_multimap(lambda g: step_size_fn(state.count) * g,
                                updates)
    return updates, ScaleByScheduleState(count=state.count + 1)

  return GradientTransformation(init_fn, update_fn)


class ScaleAndDecayByScheduleState(OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


def scale_and_decay_by_schedule(
    step_size_fn: Callable[[jnp.ndarray], jnp.ndarray],
    weight_decay: float = 0.0) -> GradientTransformation:
  """Scale updates and do weight decay using a custom schedule for `step_size`.

  Args:
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the updates by.
    weight_decay: a fixed scalar weight decay rate, which is further scaled by
      the step_size

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleAndDecayByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params):
    step_size = step_size_fn(state.count)
    updates = jax.tree_multimap(
        lambda g, p: (step_size * g - abs(step_size) * weight_decay * p),
        updates, params)
    return updates, ScaleAndDecayByScheduleState(count=state.count + 1)

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
    variance = eta / (1 + state.count) ** gamma
    all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape),
        updates, jax.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_multimap(
        lambda g, n: g + variance * n, updates, noise)
    return updates, AddNoiseState(count=state.count + 1, rng_key=all_keys[0])

  return GradientTransformation(init_fn, update_fn)


class ApplyEvery(OptState):
  """Contains a counter and a gradient accumulator."""
  count: jnp.ndarray
  grad_acc: Updates


def apply_every(k: int = 1) -> GradientTransformation:
  """Accumulate gradients and apply them every k steps.

  Args:
    k: emit non-zero gradients every k steps, otherwise accumulate them.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    grad_acc = jax.tree_multimap(jnp.zeros_like, params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state, params=None):
    del params
    c = state.count % k
    acc = c != 0
    grad_acc = jax.tree_multimap(
        lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = jax.tree_multimap(lambda ga: emit * ga, grad_acc)
    return updates, ApplyEvery(count=state.count + 1, grad_acc=grad_acc)

  return GradientTransformation(init_fn, update_fn)
