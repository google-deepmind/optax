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
"""SOAP (Second-order Optimization with Alternating Projections) optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _masking
import optax.tree

# Pytree of preconditioner matrices
PreconditionerMatrices = Any


class Preconditioner(NamedTuple):
  """Container for preconditioner matrices."""
  matrices: Tuple[Optional[jax.Array], ...]


class SoapState(NamedTuple):
  """State for the SOAP algorithm."""
  count: jax.Array
  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  gg: PreconditionerMatrices
  q: PreconditionerMatrices


def _is_preconditioner(x) -> bool:
  return isinstance(x, Preconditioner)


def _init_conditioner(
    p: jax.Array,
    max_precond_dim: int,
    precondition_1d: bool,
    dtype: jax.typing.DTypeLike,
) -> Preconditioner:
  """Initialize the preconditioner matrices for a given parameter."""
  if p.ndim == 1:
    if not precondition_1d or p.shape[0] > max_precond_dim:
      return Preconditioner((None,))
    return Preconditioner((jnp.zeros((p.shape[0], p.shape[0]), dtype=dtype),))

  return Preconditioner(tuple(
      jnp.zeros((s, s), dtype=dtype) if s <= max_precond_dim else None
      for s in p.shape
  ))


def _lerp(start: jax.Array, end: jax.Array, weight: float) -> jax.Array:
  return start + weight * (end - start)


def _update_preconditioner(
    grad: jax.Array,
    precond: Preconditioner,
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Preconditioner:
  """Update the preconditioner statistics."""
  gg = precond.matrices
  if grad.ndim == 1:
    if gg[0] is None:
      return precond
    return Preconditioner((_lerp(gg[0], jnp.outer(grad, grad), 1 - beta),))

  new_gg = []
  for idx, mat in enumerate(gg):
    if mat is None:
      new_gg.append(None)
      continue

    # Contract out all dimensions except the one we are tracking for this factor
    axes = list(range(grad.ndim))
    axes.remove(idx)

    outer_product = jnp.tensordot(
        grad,
        grad,
        axes=(axes, axes),
        precision=precision,
    )
    new_gg.append(_lerp(mat, outer_product, 1 - beta))

  return Preconditioner(tuple(new_gg))


def _project(
    grad: jax.Array,
    precond: Preconditioner,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
  """Project gradients into the eigenbasis."""
  for mat in precond.matrices:
    if mat is not None:
      grad = jnp.tensordot(
          grad,
          mat,
          axes=((0,), (0,)),
          precision=precision,
      )
    else:
      # If no preconditioner, just shift axes
      permute_order = list(range(1, grad.ndim)) + [0]
      grad = jnp.transpose(grad, axes=permute_order)

  return grad


def _project_back(
    grad: jax.Array,
    precond: Preconditioner,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
  """Project gradients back from the eigenbasis."""
  for mat in precond.matrices:
    if mat is not None:
      grad = jnp.tensordot(
          grad,
          mat,
          axes=((0,), (1,)),
          precision=precision,
      )
    else:
      grad = jnp.moveaxis(grad, 0, -1)

  return grad


def _get_orthogonal_matrix(
    gg: Optional[jax.Array], qr_dtype: jax.typing.DTypeLike
) -> Optional[jax.Array]:
  if gg is None:
    return None

  gg_mat = gg.astype(qr_dtype) if gg.dtype != qr_dtype else gg
  jitter = jnp.asarray(1e-30, dtype=qr_dtype)
  _, eigh = jnp.linalg.eigh(
      gg_mat + jitter * jnp.eye(gg_mat.shape[0], dtype=qr_dtype)
  )
  # Sort eigenvalues descending to match reference behavior
  q_mat = jnp.flip(eigh, axis=1)
  if q_mat.dtype != gg.dtype:
    q_mat = q_mat.astype(gg.dtype)
  return q_mat


def _get_orthogonal_matrix_qr(
    precond_gg: Preconditioner,
    precond_q: Preconditioner,
    exp_avg_sq: jax.Array,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    qr_dtype: jax.typing.DTypeLike = jnp.float32,
) -> Tuple[Preconditioner, jax.Array]:
  """Update the orthogonal matrices using one step of Power Iteration + QR."""
  gg = precond_gg.matrices
  q = precond_q.matrices
  final_q = []
  for ind, (m, o) in enumerate(zip(gg, q)):
    if m is None or o is None:
      final_q.append(None)
      continue

    m_mat = m.astype(qr_dtype) if m.dtype != qr_dtype else m
    o_mat = o.astype(qr_dtype) if o.dtype != qr_dtype else o

    # Estimate eigenvalues in the current basis
    est_eig = jnp.diag(
        jnp.matmul(
            jnp.matmul(o_mat.T, m_mat, precision=precision),
            o_mat,
            precision=precision,
        )
    )

    # Sort descending
    sort_idx = jnp.argsort(est_eig)[::-1]

    # Keep exp_avg_sq aligned with the sorted basis
    exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
    o_mat = o_mat[:, sort_idx]

    # Power iteration
    power_iter = jnp.matmul(m_mat, o_mat, precision=precision)
    q_new, _ = jnp.linalg.qr(power_iter)

    if q_new.dtype != m.dtype:
      q_new = q_new.astype(m.dtype)
    final_q.append(q_new)

  return Preconditioner(tuple(final_q)), exp_avg_sq


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1.0,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    qr_dtype: jax.typing.DTypeLike = jnp.float32,
) -> base.GradientTransformation:
  """Rescale updates according to the SOAP algorithm.

  SOAP (Second-order Optimization with Alternating Projections) improves
  upon Shampoo by running Adam steps in the preconditioner's eigenbasis.

  Args:
    b1: Decay rate for the exponentially weighted average of grads (Adam beta1).
    b2: Decay rate for the exponentially weighted average of squared grads
      (Adam beta2).
    shampoo_beta: Decay rate for the preconditioner statistics (L and R).
      If < 0, uses b2.
    eps: Term added to the denominator to improve numerical stability.
    precondition_frequency: How often to update the preconditioner
      orthogonal bases.
    max_precond_dim: Maximum dimension for which to construct a dense
      preconditioner. Dimensions larger than this will use Adam-style updates.
    precondition_1d: Whether to precondition 1D parameters.
    precision: JAX precision to use for matrix multiplications.
    mu_dtype: Optional `dtype` to use for moment accumulators.
    qr_dtype: `dtype` used for eigen/QR computations and
      preconditioner storage.

  Returns:
    A `GradientTransformation` object.
  """
  shampoo_beta = jnp.where(shampoo_beta >= 0, shampoo_beta, b2)
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    exp_avg = optax.tree.zeros_like(params, dtype=mu_dtype)
    exp_avg_sq = optax.tree.zeros_like(params, dtype=mu_dtype)

    gg = jax.tree.map(
        lambda p: _init_conditioner(
            p, max_precond_dim, precondition_1d, qr_dtype
        ),
        params,
    )
    q_basis = jax.tree.map(
        lambda p: _init_conditioner(
            p, max_precond_dim, precondition_1d, qr_dtype
        ),
        params,
    )

    return SoapState(
        count=jnp.zeros([], jnp.int32),
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        gg=gg,
        q=q_basis,
    )

  def _init_step(updates, state):
    new_gg = jax.tree.map(
        lambda grad, mat: _update_preconditioner(
            grad, mat, shampoo_beta, precision
        ),
        updates,
        state.gg,
        is_leaf=_is_preconditioner,
    )

    new_q = jax.tree.map(
        lambda precond: Preconditioner(
            tuple(_get_orthogonal_matrix(m, qr_dtype) for m in precond.matrices)
        ),
        new_gg,
        is_leaf=_is_preconditioner,
    )

    new_updates = optax.tree.zeros_like(updates)
    return new_updates, state._replace(gg=new_gg, q=new_q)

  def _update_step(updates, state):
    # Project gradients into the eigenbasis
    grad_projected = jax.tree.map(
        lambda grad, q_precond: _project(grad, q_precond, precision),
        updates,
        state.q,
        is_leaf=_is_preconditioner,
    )

    # Update Adam-style moments in the eigenbasis
    exp_avg = optax.tree.update_moment(grad_projected, state.exp_avg, b1, 1)
    exp_avg_sq = optax.tree.update_moment_per_elem_norm(
        grad_projected, state.exp_avg_sq, b2, 2
    )

    if mu_dtype is not None:
      exp_avg = optax.tree.cast(exp_avg, mu_dtype)
      exp_avg_sq = optax.tree.cast(exp_avg_sq, mu_dtype)

    # Bias correction
    count_inc = numerics.safe_increment(state.count)
    # The first step (count == 1) does no moment
    # updates, so effective steps is count - 1
    effective_step = count_inc - 1
    b1_correction = 1.0 - b1**effective_step
    b2_correction = 1.0 - b2**effective_step
    bias_correction = jnp.sqrt(b2_correction) / b1_correction
    # Get dtype from the first leaf of exp_avg_sq
    dtype = jax.tree.leaves(exp_avg_sq)[0].dtype
    bias_correction = bias_correction.astype(dtype)

    # Apply Adam update and project back
    norm_updates = jax.tree.map(
        lambda m, v, q_pre: _project_back(
            m / (jnp.sqrt(v) + eps), q_pre, precision
        ) * bias_correction,
        exp_avg,
        exp_avg_sq,
        state.q,
        is_leaf=_is_preconditioner,
    )

    # Update preconditioner statistics
    new_gg = jax.tree.map(
        lambda grad, mat: _update_preconditioner(
            grad, mat, shampoo_beta, precision
        ),
        updates,
        state.gg,
        is_leaf=_is_preconditioner,
    )

    # Periodic eigendecomposition update
    def _refresh_preconditioner():
      new_q_and_v = jax.tree.map(
          lambda v, g_precond, q_precond: _get_orthogonal_matrix_qr(
              g_precond, q_precond, v, precision, qr_dtype
          ),
          exp_avg_sq,
          new_gg,
          state.q,
          is_leaf=_is_preconditioner,
      )
      # Extract Preconditioner and exp_avg_sq from the results
      new_q = jax.tree.map(
          lambda x: x[0],
          new_q_and_v,
          is_leaf=lambda x: (
              isinstance(x, tuple) and isinstance(x[0], Preconditioner)
          ),
      )
      new_v = jax.tree.map(
          lambda x: x[1],
          new_q_and_v,
          is_leaf=lambda x: (
              isinstance(x, tuple) and isinstance(x[0], Preconditioner)
          ),
      )

      # Project momentum buffers to the new basis
      new_m = jax.tree.map(
          lambda m, old_q, new_q_pre: _project(
              _project_back(m, old_q, precision), new_q_pre, precision
          ),
          exp_avg,
          state.q,
          new_q,
          is_leaf=_is_preconditioner,
      )
      return new_q, new_v, new_m

    def _keep_preconditioner():
      return state.q, exp_avg_sq, exp_avg

    new_q, exp_avg_sq, exp_avg = jax.lax.cond(
        (effective_step) % precondition_frequency == 0,
        _refresh_preconditioner,
        _keep_preconditioner,
    )

    new_state = SoapState(
        count=state.count,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        gg=new_gg,
        q=new_q,
    )
    return norm_updates, new_state

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_increment(state.count)
    state = state._replace(count=count_inc)

    updates, new_state = jax.lax.cond(
        count_inc == 1,
        lambda: _init_step(updates, state),
        lambda: _update_step(updates, state),
    )
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    qr_dtype: jax.typing.DTypeLike = jnp.float32,
) -> base.GradientTransformationExtraArgs:
  """SOAP (Second-order Optimization with Alternating Projections) optimizer.

  SOAP improves upon Shampoo by running Adam steps in the preconditioner's
  eigenbasis, achieving faster convergence and better wall-clock time on LLMs.

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    shampoo_beta: Decay rate for the preconditioner statistics (L and R).
      If < 0, uses b2.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates.
    precondition_frequency: How often to update the preconditioner
      orthogonal bases.
    max_precond_dim: Maximum dimension for which to construct a dense
      preconditioner. Dimensions larger than this will use Adam-style updates.
    precondition_1d: Whether to precondition 1D parameters.
    precision: JAX precision to use for matrix multiplications.
    mu_dtype: Optional `dtype` to use for moment accumulators.
    qr_dtype: `dtype` used for eigen/QR computations and
      preconditioner storage.

  Returns:
    The corresponding `GradientTransformationExtraArgs` mapping to SOAP.

  References:
    Vyas et al., `SOAP: Improving and Stabilizing Shampoo using Adam
    <https://arxiv.org/abs/2409.11321>`_, 2024
  """
  return combine.chain(
      scale_by_soap(
          b1=b1,
          b2=b2,
          shampoo_beta=shampoo_beta,
          eps=eps,
          precondition_frequency=precondition_frequency,
          max_precond_dim=max_precond_dim,
          precondition_1d=precondition_1d,
          precision=precision,
          mu_dtype=mu_dtype,
          qr_dtype=qr_dtype,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )
