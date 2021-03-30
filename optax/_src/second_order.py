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
"""Functions for computing diagonals of Hessians & Fisher info of parameters.

Computing the Hessian or Fisher information matrices for neural networks is
typically intractible due to the quadratic memory requirements. Solving for the
diagonals of these matrices is often a better solution.

This module provides two functions for computing these diagonals, `hessian_diag`
and `fisher_diag`., each with sub-quadratic memory requirements.

"""

from typing import Any, Callable

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp


# This covers both Jax and Numpy arrays.
# TODO(b/160876114): use the pytypes defined in Chex.
Array = jnp.ndarray
# LossFun of type f(params, inputs, targets).
LossFun = Callable[[Any, Array, Array], Array]


def ravel(p: Any) -> Array:
  return ravel_pytree(p)[0]


def hvp(
    loss: LossFun,
    v: jnp.DeviceArray,
    params: Any,
    inputs: jnp.DeviceArray,
    targets: jnp.DeviceArray,
) -> jnp.DeviceArray:
  """Performs an efficient vector-Hessian (of `loss`) product.

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
  _, unravel_fn = ravel_pytree(params)
  loss_fn = lambda p: loss(p, inputs, targets)
  return jax.jvp(jax.grad(loss_fn), [params], [unravel_fn(v)])[1]


def hessian_diag(
    loss: LossFun,
    params: Any,
    inputs: jnp.DeviceArray,
    targets: jnp.DeviceArray,
) -> jnp.DeviceArray:
  """Computes the diagonal hessian of `loss` at (`inputs`, `targets`).

  Args:
    loss: the loss function.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.

  Returns:
    A DeviceArray corresponding to the product to the Hessian of `loss`
    evaluated at `(params, inputs, targets)`.
  """
  vs = jnp.eye(ravel(params).size)
  comp = lambda v: jnp.vdot(v, ravel(hvp(loss, v, params, inputs, targets)))
  return jax.vmap(comp)(vs)


def fisher_diag(
    negative_log_likelihood: LossFun,
    params: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.DeviceArray:
  """Computes the diagonal of the (observed) Fisher information matrix.

  Args:
    negative_log_likelihood: the negative log likelihood function.
    params: model parameters.
    inputs: inputs at which `negative_log_likelihood` is evaluated.
    targets: targets at which `negative_log_likelihood` is evaluated.

  Returns:
    An Array corresponding to the product to the Hessian of
    `negative_log_likelihood` evaluated at `(params, inputs, targets)`.
  """
  return jnp.square(
      ravel(jax.grad(negative_log_likelihood)(params, inputs, targets)))
