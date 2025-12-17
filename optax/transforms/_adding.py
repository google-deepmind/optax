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
"""Additive components in gradient transformations."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union
import warnings

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src import wrappers
import optax.tree


class WeightDecaySchedule(NamedTuple):
  """Maintains count for weight decay scheduling."""
  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32


def add_decayed_weights(
    weight_decay: base.ScalarOrSchedule = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    del params
    if callable(weight_decay):
      return WeightDecaySchedule(count=jnp.zeros([], jnp.int32))
    else:
      return base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    if callable(weight_decay):
      new_state = WeightDecaySchedule(numerics.safe_increment(state.count))
    else:
      new_state = state

    # If weight decay is a zero constant, we can skip the update.
    if isinstance(weight_decay, (int, float)) and weight_decay == 0.0:
      return updates, new_state

    s = weight_decay(state.count) if callable(weight_decay) else weight_decay
    updates = jax.tree.map(
        lambda g, p: None if g is None else g + s * p,
        updates,
        params,
        is_leaf=lambda x: x is None,
    )
    return updates, new_state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask
    )
  return base.GradientTransformation(init_fn, update_fn)


class AddNoiseState(NamedTuple):
  """State for adding gradient noise. Contains a count for annealing."""

  count: jax.Array
  rng_key: jax.Array


def add_noise(
    eta: jax.typing.ArrayLike,
    gamma: jax.typing.ArrayLike,
    key: jax.Array | int | None = None,
    *,
    seed: int | None = None,  # deprecated
) -> base.GradientTransformation:
  """Add gradient noise.

  Args:
    eta: Base variance of the gaussian noise added to the gradient.
    gamma: Decay exponent for annealing of the variance.
    key: random generator key for noise generation.
    seed: deprecated, use key instead.

  Returns:
    A :class:`optax.GradientTransformation` object.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> key = jax.random.key(0)  # could also be key=0
    >>> noise = optax.add_noise(eta=0.01, gamma=0.55, key=key)
    >>> sgd = optax.scale_by_learning_rate(learning_rate=0.003)
    >>> solver = optax.chain(noise, sgd)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Neelakantan et al, `Adding Gradient Noise Improves Learning for Very Deep
    Networks <https://arxiv.org/abs/1511.06807>`_, 2015
  """

  if seed is not None:
    warnings.warn(
        '"seed" is deprecated and will be removed in optax 0.2.7, use "key".',
        DeprecationWarning,
    )
    if key is not None:
      raise ValueError('Only one of seed or key can be specified.')
    key = seed
  if key is None:
    warnings.warn('Specifying a key will be required in optax 0.2.7.')
    key = 0

  def init_fn(params):
    del params
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=utils.canonicalize_key(key)
    )

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_increment(state.count)
    standard_deviation = jnp.sqrt(eta / count_inc**gamma)

    rng_key, sample_key = jax.random.split(state.rng_key)
    noise = optax.tree.random_like(
        sample_key, target_tree=updates, sampler=jax.random.normal
    )
    updates = optax.tree.add_scale(
        tree_x=updates, scalar=standard_deviation, tree_y=noise
    )
    return updates, AddNoiseState(count=count_inc, rng_key=rng_key)

  return base.GradientTransformation(init_fn, update_fn)
