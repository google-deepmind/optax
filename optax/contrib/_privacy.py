# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Differential Privacy utilities."""

from typing import NamedTuple, Optional
import warnings

import jax

from optax import transforms

from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils


class DifferentiallyPrivateAggregateState(NamedTuple):
  """State containing PRNGKey for `differentially_private_aggregate`."""

  rng_key: jax.Array


def differentially_private_aggregate(
    l2_norm_clip: jax.typing.ArrayLike,
    noise_multiplier: jax.typing.ArrayLike,
    key: jax.Array | int | None = None,
    *,
    seed: int | None = None,  # deprecated
) -> base.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    key: random generator key for noise generation.
    seed: deprecated, use key instead.

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    Unlike other transforms, `differentially_private_aggregate` expects
    the input updates to have a batch dimension in the 0th axis. That is, this
    function expects per-example gradients as input (which are easy to obtain in
    JAX using `jax.vmap`). It can still be composed with other transformations
    as long as it is the first in the chain.

  .. warning::
    Generic gradient aggregation tools like :class:`optax.MultiSteps` or
    :func:`optax.apply_every` won't work correctly with this transformation
    since the whole point of this transformation is to aggregate gradients in a
    specific way.
  """

  if seed is not None:
    warnings.warn(
        '"seed" is deprecated and will be removed in optax 0.2.7, use "key".',
        DeprecationWarning,
    )
    if key is not None:
      raise ValueError('Only one of seed or key can be specified.')
    key = jax.random.key(seed)
  if key is None:
    warnings.warn('Specifying a key will be required in optax 0.2.7.')
    key = jax.random.key(0)
  key = utils.canonicalize_key(key)

  noise_std = l2_norm_clip * noise_multiplier

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=key)

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree.flatten(updates)
    bsize = grads_flat[0].shape[0]
    clipped, _ = transforms.per_example_global_norm_clip(
        grads_flat,
        l2_norm_clip
    )

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [
        (g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
        for g, r in zip(clipped, rngs)
    ]
    return (
        jax.tree.unflatten(grads_treedef, noised),
        DifferentiallyPrivateAggregateState(rng_key=new_key),
    )

  return base.GradientTransformation(init_fn, update_fn)


def dpsgd(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: jax.typing.ArrayLike,
    noise_multiplier: jax.typing.ArrayLike,
    seed: int,
    momentum: Optional[jax.typing.ArrayLike] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The DPSGD optimizer.

  Differential privacy is a standard for privacy guarantees of algorithms
  learning from aggregate databases including potentially sensitive information.
  DPSGD offers protection against a strong adversary with full knowledge of the
  training mechanism and access to the model's parameters.

  Args:
    learning_rate: A fixed global scaling factor.
    l2_norm_clip: Maximum L2 norm of the per-example gradients.
    noise_multiplier: Ratio of standard deviation to the clipping norm.
    seed: Initial seed used for the jax.random.PRNGKey
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    This :class:`optax.GradientTransformation` expects input updates to have a
    batch dimension on the 0th axis. That is, this function expects per-example
    gradients as input (which are easy to obtain in JAX using `jax.vmap`).

  .. warning::
    Generic gradient aggregation tools like :class:`optax.MultiSteps` or
    :func:`optax.apply_every` won't work correctly with this transformation
    since the whole point of this transformation is to aggregate gradients in a
    specific way.
  """
  return combine.chain(
      differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed,
      ),
      (
          transforms.trace(decay=momentum, nesterov=nesterov)
          if momentum is not None
          else base.identity()
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
