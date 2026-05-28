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
"""Deprecated second order utilities kept for backward compatibility.
"""

import abc
import functools
from typing import Any, Protocol

import jax
from jax import flatten_util
import jax.numpy as jnp
from optax._src.deprecations import warn_deprecated_function  # pylint: disable=g-importing-member


def _ravel(p: Any) -> jax.Array:
  return flatten_util.ravel_pytree(p)[0]


class LossFn(Protocol):
  """A loss function to be optimized."""

  @abc.abstractmethod
  def __call__(
      self, params: Any, inputs: jax.Array, targets: jax.Array
  ) -> jax.Array:
    ...


@functools.partial(warn_deprecated_function, version_removed='0.2.9')
def hvp(
    loss: LossFn,
    v: jax.Array,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Performs an efficient vector-Hessian (of `loss`) product.

  .. deprecated: 0.2.7. This function will be removed in 0.2.9

  Args:
    loss: the loss function.
    v: a vector of size `ravel(params)`.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.

  Returns:
    An Array corresponding to the product of `v` and the Hessian of `loss`
    evaluated at `(params, inputs, targets)`.
  """
  _, unravel_fn = flatten_util.ravel_pytree(params)
  loss_fn = lambda p: loss(p, inputs, targets)
  return jax.jvp(jax.grad(loss_fn), [params], [unravel_fn(v)])[1]


@functools.partial(warn_deprecated_function, version_removed='0.2.9')
def hessian_diag(
    loss: LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Computes the diagonal hessian of `loss` at (`inputs`, `targets`).

  .. deprecated: 0.2.7. This function will be removed in 0.2.9

  Args:
    loss: the loss function.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.

  Returns:
    A DeviceArray corresponding to the product to the Hessian of `loss`
    evaluated at `(params, inputs, targets)`.
  """
  vs = jnp.eye(_ravel(params).size)
  comp = lambda v: jnp.vdot(v, _ravel(hvp(loss, v, params, inputs, targets)))
  return jax.vmap(comp)(vs)


@functools.partial(warn_deprecated_function, version_removed='0.2.9')
def fisher_diag(
    negative_log_likelihood: LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Computes the diagonal of the (observed) Fisher information matrix.

  .. deprecated: 0.2.7. This function will be removed in 0.2.9

  Args:
    negative_log_likelihood: the negative log likelihood function with expected
      signature `loss = fn(params, inputs, targets)`.
    params: model parameters.
    inputs: inputs at which `negative_log_likelihood` is evaluated.
    targets: targets at which `negative_log_likelihood` is evaluated.

  Returns:
    An Array corresponding to the product to the Hessian of
    `negative_log_likelihood` evaluated at `(params, inputs, targets)`.
  """
  return jnp.square(
      _ravel(jax.grad(negative_log_likelihood)(params, inputs, targets))
  )
