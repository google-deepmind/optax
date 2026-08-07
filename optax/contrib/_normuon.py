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
"""NorMuon optimizer."""

import math
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.contrib._muon import _DEFAULT_NS_COEFFS
from optax.contrib._muon import _is_weight_dim_nums
from optax.contrib._muon import _NS_COEFFS_PRESET_DICT
from optax.contrib._muon import MuonDimensionNumbers
from optax.contrib._muon import orthogonalize_via_newton_schulz
from optax.contrib._muon import scale_by_shape
from optax.contrib._muon import WeightDimNumOrFn
from optax.transforms import _masking
import optax.tree


class NorMuonState(NamedTuple):
  """State for the NorMuon algorithm."""
  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates
  ns_coeffs: jax.typing.ArrayLike


def scale_by_normuon(
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
              jax.typing.ArrayLike],
        tuple[
            tuple[
                jax.typing.ArrayLike, jax.typing.ArrayLike,
                jax.typing.ArrayLike
            ],
            ...,
        ],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    preconditioning: Literal[
        'frobenius', 'spectral', 'aol', 'schatten'
    ] = 'frobenius',
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    normuon_scale: jax.typing.ArrayLike = 0.2,
) -> base.GradientTransformation:
  r"""Rescale updates according to the NorMuon algorithm.

  NorMuon extends Muon with row-wise adaptive normalization after the
  Newton-Schulz orthogonalization step. This balances neuron utilization
  with negligible memory overhead compared to Muon.

  Args:
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if ``ns_coeffs`` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    beta2: Decay rate for the row-wise second moment estimates.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    preconditioning: Which preconditioning method to use before NS iterations.
    weight_dimension_numbers: An optional tree with the same structure as the
      params of ``MuonDimensionNumbers``s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    normuon_scale: Adaptive learning rate coefficient (default 0.2).

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Li et al., `NorMuon: Making Muon more efficient and scalable
    <https://arxiv.org/abs/2510.05491>`_, 2025
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)
    # nu stores row-wise second moments: shape (m,) for a (m, n) param.
    nu = jax.tree.map(lambda x: jnp.zeros(x.shape[:-1], dtype=mu_dtype),
                      params)
    ns_coeffs_ = jnp.asarray(ns_coeffs)

    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    if ns_coeffs_.ndim == 2:
      if ns_coeffs_.shape[0] > ns_steps:
        raise ValueError(f'Not enough coeffs to perform {ns_steps} steps')
      ns_coeffs_ = ns_coeffs_[-ns_steps:]

    return NorMuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        nu=nu,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    if callable(weight_dimension_numbers):
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    mu = optax.tree.update_moment(updates, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(
              mu, beta, numerics.safe_increment(count_inc)
          ),
          optax.tree.bias_correction(updates, beta, count_inc),
      )
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

    # Apply Newton-Schulz orthogonalization.
    ortho = jax.tree.map(
        lambda x, dim_num: orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, preconditioning, eps, dim_num),
        mu_hat, resolved_weight_dim_nums, is_leaf=_is_weight_dim_nums)

    # Row-wise second moment tracking.
    def _update_nu(o, nu_prev):
      row_sq = jnp.mean(o ** 2, axis=-1)
      return beta2 * nu_prev + (1 - beta2) * row_sq

    new_nu = jax.tree.map(_update_nu, ortho, state.nu)

    # Row-wise normalization and adaptive scaling (paper Algorithm 1).
    def _normalize(o, nu_new):
      o_hat = o / (jnp.sqrt(nu_new[..., None]) + eps)
      m_n = math.prod(o.shape[-2:]) if o.ndim >= 2 else o.shape[-1]
      frob = jnp.linalg.norm(o_hat, ord='fro')
      scale = normuon_scale * jnp.sqrt(m_n) / (frob + eps)
      return o_hat * scale

    new_updates = jax.tree.map(_normalize, ortho, new_nu)

    mu = optax.tree.cast(mu, mu_dtype)
    return new_updates, NorMuonState(
        count=count_inc,
        mu=mu,
        nu=new_nu,
        ns_coeffs=state.ns_coeffs,
    )

  return base.GradientTransformation(init_fn, update_fn)


def normuon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
              jax.typing.ArrayLike],
        tuple[
            tuple[
                jax.typing.ArrayLike, jax.typing.ArrayLike,
                jax.typing.ArrayLike
            ],
            ...,
        ],
        str,
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    preconditioning: Literal[
        'frobenius', 'spectral', 'aol', 'schatten'
    ] = 'frobenius',
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    normuon_scale: jax.typing.ArrayLike = 0.2,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  r"""NorMuon: Muon with row-wise adaptive normalization.

  NorMuon extends the Muon optimizer with row-wise adaptive normalization
  applied after Newton-Schulz orthogonalization. This ensures balanced
  neuron utilization with negligible memory overhead compared to Muon.

  Like Muon, NorMuon is only defined for 2D parameters (matrices). Non-2D
  parameters are passed through an AdamW optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-Schulz method (can be a string
      indicator for a preset). Existing presets: ``muon``, ``dion``.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if ``ns_coeffs`` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    beta2: Decay rate for the row-wise second moment estimates.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization.
    weight_decay_mask: A tree with same structure as (or a prefix of) the
      params PyTree, or a Callable that returns such a pytree given the
      params/updates. The leaves should be booleans, ``True`` for
      leaves/subtrees you want to apply the weight decay to, and ``False``
      for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    preconditioning: Which preconditioning method to use before NS iterations.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adam_learning_rate: Auxiliary learning rate for the Adam optimizer.
      If ``None``, the learning rate for Adam defaults to the same as NorMuon.
    muon_weight_dimension_numbers: An optional tree of
      ``MuonDimensionNumbers``s, specifying how to reshape the parameters for
      orthogonalization. A ``None`` value indicates that the parameter is not
      a NorMuon parameter and will be optimized with Adam. If not provided,
      NorMuon is applied to all 2D parameters.
    normuon_scale: Adaptive learning rate coefficient (default 0.2).
    consistent_rms: An optional float to activate consistent RMS scaling.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  References:
    Li et al., `NorMuon: Making Muon more efficient and scalable
    <https://arxiv.org/abs/2510.05491>`_, 2025
  """

  if adam_learning_rate is None:
    adam_learning_rate = learning_rate

  if isinstance(ns_coeffs, str):
    if ns_coeffs not in _NS_COEFFS_PRESET_DICT:
      raise ValueError(f'Unknown ns_coeff preset string: {ns_coeffs}')
    ns_coeffs_ = _NS_COEFFS_PRESET_DICT[ns_coeffs]
  else:
    ns_coeffs_ = ns_coeffs

  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'normuon' if x.ndim == 2 else 'adam', params
    )
    muon_weight_dimension_numbers = MuonDimensionNumbers()
  else:
    def param_labels(params):
      dim_nums = (muon_weight_dimension_numbers(params)
                  if callable(muon_weight_dimension_numbers)
                  else muon_weight_dimension_numbers)
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda y: 'normuon' if dim_num is not None else 'adam', x)
      return jax.tree.map(
          populate_subtree_, dim_nums, params,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x))

  def muon_weight_dim_nums_fn(params):
    dim_nums = (muon_weight_dimension_numbers(params)
                if callable(muon_weight_dimension_numbers)
                else muon_weight_dimension_numbers)
    mask = jax.tree.map(
        lambda label: label == 'normuon', param_labels(params))
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x)
                         or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask)
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          'normuon': combine.chain(
              scale_by_normuon(
                  ns_coeffs=ns_coeffs_,
                  ns_steps=ns_steps,
                  beta=beta,
                  beta2=beta2,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  preconditioning=preconditioning,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  normuon_scale=normuon_scale,
              ),
              scale_by_shape(
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=adam_learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
