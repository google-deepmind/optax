# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""AdaHessian optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree
from optax.contrib import _hutchinson


def hutchinson_estimator_diag_hessian(
    random_seed: Optional[jax.Array] = None,
    n_samples: int = 1,
):
  """Returns a GradientTransformationExtraArgs computing the Hessian diagonal.

  The Hessian diagonal is estimated using Hutchinson's estimator, which is
  unbiased but has high variance. Multiple samples reduce variance.

  Args:
    random_seed: key used to generate random vectors.
    n_samples: number of Hutchinson samples to average over per update.

  Returns:
    GradientTransformationExtraArgs
  """
  return _hutchinson.hutchinson_estimator_diag_hessian(
      random_seed=random_seed, n_samples=n_samples
  )


class AdaHessianState(NamedTuple):
  """State for the AdaHessian optimizer."""

  count: jax.Array  # shape=(), dtype=jnp.int32
  mu: base.Updates  # momentum
  nu: base.Updates  # EMA of squared Hessian diagonal
  hessian_diag: base.Updates  # cached Hessian diagonal
  hessian_fn_state: Any


def _average_conv_kernel_hessian(hessian_diag):
  """Average per-kernel Hessian values across spatial dims for conv weights."""
  def maybe_average(h):
    if h.ndim == 4:
      mean = jnp.mean(jnp.abs(h), axis=(2, 3), keepdims=True)
      return jnp.ones_like(h) * mean
    return h

  return jax.tree.map(maybe_average, hessian_diag)


def scale_by_adahessian(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    hessian_power: jax.typing.ArrayLike = 1.0,
    update_interval: jax.typing.ArrayLike = 1,
    average_conv_kernel: bool = True,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
  """Rescale updates according to the AdaHessian algorithm.

  A separate GradientTransformation is required through the argument
  `hessian_diagonal_fn` to compute the diagonal of the Hessian. Any extra
  arguments required by the hessian_diagonal_fn's update function can be
  passed through the update function as trailing keyword arguments
  (**kwargs). The default hessian_diagonal_fn is Hutchinson's estimator and
  needs the objective function as an extra argument, `obj_fn`. `obj_fn` must
  accept `params` as its only argument and return only a scalar (the loss).
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  hessian_diagonal_fn = base.with_extra_args_support(hessian_diagonal_fn)

  def init_fn(params):
    return AdaHessianState(
        count=jnp.zeros([], jnp.int32),
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        nu=optax.tree.zeros_like(params),
        hessian_diag=optax.tree.zeros_like(params),
        hessian_fn_state=hessian_diagonal_fn.init(params),
    )

  def update_fn(updates, state: AdaHessianState, params=None, **hess_fn_kwargs):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    count_inc = numerics.safe_int32_increment(state.count)

    # First-moment (momentum) estimate of gradients.
    mu = optax.tree.update_moment(updates, state.mu, b1, 1)
    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)

    def update_hessian_diag(hess_fn_state, hessian_diag):
      # Use the provided Hessian-diagonal estimator (default: Hutchinson).
      hessian_diag, hess_fn_state = hessian_diagonal_fn.update(
          updates, hess_fn_state, params=params, **hess_fn_kwargs
      )
      if average_conv_kernel:
        # Stabilize conv kernels by averaging across spatial dimensions.
        hessian_diag = _average_conv_kernel_hessian(hessian_diag)
      return hess_fn_state, hessian_diag

    # Recompute the Hessian diagonal periodically, otherwise reuse the cached
    # one.
    hessian_fn_state, hessian_diag = jax.lax.cond(
        jnp.equal(state.count % update_interval, 0),
        update_hessian_diag,
        lambda h, d: (h, d),
        state.hessian_fn_state,
        state.hessian_diag,
    )

    # EMA of squared Hessian diagonal (per-parameter scaling term).
    nu = optax.tree.update_moment_per_elem_norm(hessian_diag, state.nu, b2, 2)
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)

    # Scale the momentum by the Hessian power and epsilon for numerical safety.
    denom = jax.tree.map(
        lambda n: jnp.power(n, hessian_power / 2) + eps, nu_hat
    )
    updates = jax.tree.map(lambda m, d: m / d, mu_hat, denom)

    mu = optax.tree.cast(mu, mu_dtype)

    new_state = AdaHessianState(
        count=count_inc,
        mu=mu,
        nu=nu,
        hessian_diag=hessian_diag,
        hessian_fn_state=hessian_fn_state,
    )
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def adahessian(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    hessian_power: jax.typing.ArrayLike = 1.0,
    update_interval: jax.typing.ArrayLike = 1,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    average_conv_kernel: bool = True,
    hessian_diagonal_fn: Union[
        base.GradientTransformation,
        base.GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
  """AdaHessian optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the Hessian diagonal estimates.
    eps: Small constant to avoid division by zero.
    hessian_power: Exponent on the Hessian diagonal EMA (k in the paper).
    update_interval: Interval for updating the Hessian diagonal estimate.
    weight_decay: Strength of decoupled weight decay regularization.
    weight_decay_mask: mask controlling which params receive weight decay.
    average_conv_kernel: If True, average Hessian values across spatial
      dimensions for 4D convolution kernels.
    hessian_diagonal_fn: GradientTransformation that computes the diagonal of
      the Hessian. Default is Hutchinson's estimator.
    mu_dtype: dtype of the first moment estimates.

  Returns:
    optax.GradientTransformationExtraArgs
  """
  return combine.chain(
      scale_by_adahessian(
          b1=b1,
          b2=b2,
          eps=eps,
          hessian_power=hessian_power,
          update_interval=update_interval,
          average_conv_kernel=average_conv_kernel,
          hessian_diagonal_fn=hessian_diagonal_fn,
          mu_dtype=mu_dtype,
      ),
      transform.add_decayed_weights(weight_decay, weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate),
  )
