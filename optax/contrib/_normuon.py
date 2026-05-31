# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""NorMuon.

Implementation of NorMuon (Neuron-wise Normalized Muon), proposed in:
https://arxiv.org/abs/2510.05491

NorMuon augments Muon with a neuron-wise second-moment statistic computed from
orthogonalized updates, and uses this to normalize update magnitudes across
neurons.
"""

import math
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils
from optax.contrib import _muon
from optax.transforms import _masking
import optax.tree

MuonDimensionNumbers = _muon.MuonDimensionNumbers
WeightDimNumOrFn = _muon.WeightDimNumOrFn


_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _v_shape(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[int, int]:
  """Shape of NorMuon per-neuron accumulator for a tensor and dim spec.

  We reshape a tensor to (batch, reduction, output)
  (see `_muon._compute_muon_reshape`) and store a second-moment accumulator
  per (batch, output) entry.
  """
  reduction_axes, output_axes = _muon._normalize_axes(x, dim_nums)  # pylint: disable=protected-access
  batch_axes = tuple(
      sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes))
  )
  batch_size = math.prod(x.shape[ax] for ax in batch_axes) if batch_axes else 1
  output_size = math.prod(x.shape[ax] for ax in output_axes)
  return batch_size, output_size


def _broadcast_dim_nums(
    tree: base.PyTree,
    dim_nums: MuonDimensionNumbers | None,
) -> base.PyTree:
  """Broadcast a single dim spec (or None) to match a pytree structure."""
  return jax.tree.map(lambda _: dim_nums, tree)


def _resolve_dim_nums(
    tree: base.PyTree,
    weight_dimension_numbers: WeightDimNumOrFn | None,
) -> base.PyTree:
  """Resolve weight dimension numbers to a pytree matching `tree` structure."""
  if callable(weight_dimension_numbers):
    resolved = weight_dimension_numbers(tree)
  else:
    resolved = weight_dimension_numbers

  # A single MuonDimensionNumbers (or None) applies to all leaves.
  if resolved is None or _is_weight_dim_nums(resolved):
    return _broadcast_dim_nums(tree, resolved)
  return resolved


class NorMuonState(NamedTuple):
  """State for the NorMuon algorithm."""

  mu: base.Updates
  v: base.Updates
  ns_coeffs: jax.Array  # shape=(3,) or (n, 3)


class _NormuonUpdateAndV(NamedTuple):
  update: jax.Array
  v: jax.Array


def scale_by_normuon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    rms_scale: float = 0.2,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    v_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the NorMuon algorithm.

  NorMuon applies Muon's Newton-Schulz orthogonalization to a momentum matrix,
  then computes a per-neuron (output-axis) second moment from the orthogonalized
  update and uses it to normalize update magnitudes across neurons.

  The normalization is applied per (batch, output) block of a weight tensor,
  where "batch" are any axes not listed in `MuonDimensionNumbers`, and "output"
  are the axes listed in `output_axis`. This matches Algorithm 1 in
  https://arxiv.org/abs/2510.05491 up to a possible transpose depending on the
  chosen `MuonDimensionNumbers`.

  Args:
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the neuron-wise second moment accumulator.
    eps: Term added inside square roots to improve numerical stability.
    rms_scale: Target RMS of each orthogonalized-and-normalized update matrix.
    mu_dtype: Data type of the momentum accumulator.
    v_dtype: Data type of the neuron-wise second moment accumulator.
    weight_dimension_numbers: An optional tree with the same structure as the
      params, specifying how to reshape tensors before/after orthogonalization.
      A callable may be provided to generate this tree from params/updates.
      `None` implies that all parameters are 2D matrices.

  Returns:
    A `GradientTransformation` object.
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  v_dtype = jnp.float32 if v_dtype is None else v_dtype
  v_dtype = utils.canonicalize_dtype(v_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )

    resolved_dim_nums = _resolve_dim_nums(params, weight_dimension_numbers)

    def _init_v_leaf(p: jax.Array, dim_num: MuonDimensionNumbers | None):
      if dim_num is None:
        if p.ndim != 2:
          raise ValueError(
              'NorMuon requires `weight_dimension_numbers` for non-2D tensors,'
              f' got rank={p.ndim} and {dim_num=}.'
          )
        dim_num = MuonDimensionNumbers()
      batch_size, output_size = _v_shape(p, dim_num)
      return jnp.zeros((batch_size, output_size), dtype=v_dtype)

    v = jax.tree.map(
        _init_v_leaf, params, resolved_dim_nums, is_leaf=_is_weight_dim_nums
    )

    return NorMuonState(mu=mu, v=v, ns_coeffs=ns_coeffs_)

  def update_fn(updates, state, params=None):
    del params
    resolved_dim_nums = _resolve_dim_nums(updates, weight_dimension_numbers)

    mu = optax.tree.update_moment(updates, state.mu, b1, 1)

    # Muon orthogonalization.
    ortho = jax.tree.map(
        lambda x, dim_num: _muon.orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, eps=eps, dimension_numbers=dim_num
        ),
        mu,
        resolved_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )

    # NorMuon neuron-wise normalization + per-matrix RMS scaling.
    def _normalize_leaf(
        o: jax.Array,
        v: jax.Array,
        dim_num: MuonDimensionNumbers | None,
    ):
      if dim_num is None:
        if o.ndim != 2:
          raise ValueError(
              'NorMuon requires `weight_dimension_numbers` for non-2D tensors,'
              f' got rank={o.ndim} and {dim_num=}.'
          )
        dim_num = MuonDimensionNumbers()

      reshape_fn, inverse_fn = _muon._compute_muon_reshape(o, dim_num)  # pylint: disable=protected-access
      o_flat = reshape_fn(o)  # (batch, reduction, output)

      mean_sq = jnp.mean(jnp.square(o_flat), axis=1)  # (batch, output)
      v_new = b2 * v + (1.0 - b2) * mean_sq

      denom = jnp.sqrt(v_new[:, None, :] + eps)
      o_norm = o_flat / denom

      rms = jnp.sqrt(jnp.mean(jnp.square(o_norm), axis=(1, 2)))
      scale = rms_scale / (rms + eps)
      o_scaled = o_norm * scale[:, None, None]

      return _NormuonUpdateAndV(inverse_fn(o_scaled), v_new)

    updates_and_v = jax.tree.map(
        _normalize_leaf,
        ortho,
        state.v,
        resolved_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )
    _is_update_and_v = lambda x: isinstance(x, _NormuonUpdateAndV)
    updates = jax.tree.map(
        lambda uv: uv.update, updates_and_v, is_leaf=_is_update_and_v
    )
    v = jax.tree.map(
        lambda uv: uv.v, updates_and_v, is_leaf=_is_update_and_v
    )

    mu = optax.tree.cast(mu, mu_dtype)
    v = optax.tree.cast(v, v_dtype)
    return updates, NorMuonState(mu=mu, v=v, ns_coeffs=state.ns_coeffs)

  return base.GradientTransformation(init_fn, update_fn)


def normuon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    rms_scale: float = 0.2,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    v_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    normuon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  r"""NorMuon: Neuron-wise Normalized Muon.

  NorMuon applies Muon-style orthogonalization to momentum matrices for 2D
  parameters (or parameters specified via `normuon_weight_dimension_numbers`),
  and then normalizes orthogonalized updates using neuron-wise second moments.

  Non-NorMuon parameters are optimized with AdamW.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the neuron-wise second moment accumulator.
    eps: Term added inside square roots to improve numerical stability.
    rms_scale: Target RMS of each orthogonalized-and-normalized update matrix.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the first moment accumulator.
    v_dtype: Data type of the neuron-wise second moment accumulator.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    normuon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`
      specifying how to reshape parameters for orthogonalization. A `None` leaf
      indicates that the parameter will be optimized with AdamW. If not
      provided, NorMuon is applied to all 2D parameters.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Li et al., `NorMuon: Making Muon more efficient and scalable
    <https://arxiv.org/abs/2510.05491>`_, 2025
  """
  if normuon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'normuon' if x.ndim == 2 else 'adam', params
    )
    normuon_weight_dimension_numbers = MuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          normuon_weight_dimension_numbers(params)
          if callable(normuon_weight_dimension_numbers)
          else normuon_weight_dimension_numbers
      )

      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda _: 'normuon' if dim_num is not None else 'adam', x
      )
      return jax.tree.map(
          populate_subtree_,
          dim_nums,
          params,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
      )

  def normuon_weight_dim_nums_fn(params):
    dim_nums = (
        normuon_weight_dimension_numbers(params)
        if callable(normuon_weight_dimension_numbers)
        else normuon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == 'normuon', param_labels(params))
    is_leaf = lambda x: (
        x is None
        or _is_weight_dim_nums(x)
        or isinstance(x, _masking.MaskedNode)
    )
    populate_subtree_ = lambda dim_num, submask: jax.tree.map(
        lambda m: dim_num if m else _masking.MaskedNode(), submask
    )
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          'normuon': combine.chain(
              scale_by_normuon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  b1=b1,
                  b2=b2,
                  eps=eps,
                  rms_scale=rms_scale,
                  mu_dtype=mu_dtype,
                  v_dtype=v_dtype,
                  weight_dimension_numbers=normuon_weight_dim_nums_fn,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
          ),
      },
      param_labels=param_labels,
  )
