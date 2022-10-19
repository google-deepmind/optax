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
"""Linear algebra utilities used in optimisation."""

import chex
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from optax._src import base
from optax._src import numerics


def global_norm(updates: base.Updates) -> base.Updates:
  """Compute the global norm across a nested structure of tensors."""
  return jnp.sqrt(sum(
      jnp.sum(numerics.abs_sq(x)) for x in jax.tree_util.tree_leaves(updates)))


def power_iteration(matrix: chex.Array,
                    num_iters: int = 100,
                    error_tolerance: float = 1e-6,
                    precision: lax.Precision = lax.Precision.HIGHEST):
  r"""Power iteration algorithm.

  The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
  a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
  of `A`, and a vector v, which is the corresponding eigenvector of `A`.

  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)

  Args:
    matrix: the symmetric PSD matrix.
    num_iters: Number of iterations.
    error_tolerance: Iterative exit condition.
    precision: precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise);
      b) lax.Precision.HIGH (increased precision, slower);
      c) lax.Precision.HIGHEST (best possible precision, slowest).

  Returns:
    eigen vector, eigen value
  """
  matrix_size = matrix.shape[-1]
  def _iter_condition(state):
    i, unused_v, unused_s, unused_s_v, run_step = state
    return jnp.logical_and(i < num_iters, run_step)

  def _iter_body(state):
    """One step of power iteration."""
    i, new_v, s, s_v, unused_run_step = state
    new_v = new_v / jnp.linalg.norm(new_v)

    s_v = jnp.einsum('ij,j->i', matrix, new_v, precision=precision)
    s_new = jnp.einsum('i,i->', new_v, s_v, precision=precision)
    return (i + 1, s_v, s_new, s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance))

  # Figure out how to use step as seed for random.
  v_0 = np.random.uniform(-1.0, 1.0, matrix_size).astype(matrix.dtype)

  init_state = tuple([0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True])
  _, v_out, s_out, _, _ = lax.while_loop(
      _iter_condition, _iter_body, init_state)
  v_out = v_out / jnp.linalg.norm(v_out)
  return v_out, s_out


def matrix_inverse_pth_root(matrix: chex.Array,
                            p: int,
                            num_iters: int = 100,
                            ridge_epsilon: float = 1e-6,
                            error_tolerance: float = 1e-6,
                            precision: lax.Precision = lax.Precision.HIGHEST):
  """Computes `matrix^(-1/p)`, where `p` is a positive integer.

  This function uses the Coupled newton iterations algorithm for
  the computation of a matrix's inverse pth root.


  References:
    [Functions of Matrices, Theory and Computation,
     Nicholas J Higham, Pg 184, Eq 7.18](
     https://epubs.siam.org/doi/book/10.1137/1.9780898717778)

  Args:
    matrix: the symmetric PSD matrix whose power it to be computed
    p: exponent, for p a positive integer.
    num_iters: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
    error_tolerance: Error indicator, useful for early termination.
    precision: precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise);
      b) lax.Precision.HIGH (increased precision, slower);
      c) lax.Precision.HIGHEST (best possible precision, slowest).

  Returns:
    matrix^(-1/p)
  """

  # We use float32 for the matrix inverse pth root.
  # Switch to f64 if you have hardware that supports it.
  matrix_size = matrix.shape[0]
  alpha = jnp.asarray(-1.0 / p, jnp.float32)
  identity = jnp.eye(matrix_size, dtype=jnp.float32)
  _, max_ev = power_iteration(
      matrix=matrix, num_iters=100,
      error_tolerance=1e-6, precision=precision)
  ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-16)

  def _unrolled_mat_pow_1(mat_m):
    """Computes mat_m^1."""
    return mat_m

  def _unrolled_mat_pow_2(mat_m):
    """Computes mat_m^2."""
    return jnp.matmul(mat_m, mat_m, precision=precision)

  def _unrolled_mat_pow_4(mat_m):
    """Computes mat_m^4."""
    mat_pow_2 = _unrolled_mat_pow_2(mat_m)
    return jnp.matmul(
        mat_pow_2, mat_pow_2, precision=precision)

  def _unrolled_mat_pow_8(mat_m):
    """Computes mat_m^4."""
    mat_pow_4 = _unrolled_mat_pow_4(mat_m)
    return jnp.matmul(
        mat_pow_4, mat_pow_4, precision=precision)

  def mat_power(mat_m, p):
    """Computes mat_m^p, for p == 1, 2, 4 or 8.

    Args:
      mat_m: a square matrix
      p: a positive integer

    Returns:
      mat_m^p
    """
    # We unrolled the loop for performance reasons.
    exponent = jnp.round(jnp.log2(p))
    return lax.switch(
        jnp.asarray(exponent, jnp.int32), [
            _unrolled_mat_pow_1,
            _unrolled_mat_pow_2,
            _unrolled_mat_pow_4,
            _unrolled_mat_pow_8,
        ], (mat_m))

  def _iter_condition(state):
    (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error,
     run_step) = state
    error_above_threshold = jnp.logical_and(
        error > error_tolerance, run_step)
    return jnp.logical_and(i < num_iters, error_above_threshold)

  def _iter_body(state):
    (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=precision)
    new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=precision)
    new_error = jnp.max(jnp.abs(new_mat_m - identity))
    # sometimes error increases after an iteration before decreasing and
    # converging. 1.2 factor is used to bound the maximal allowed increase.
    return (i + 1, new_mat_m, new_mat_h, mat_h, new_error,
            new_error < error * 1.2)

  if matrix_size == 1:
    resultant_mat_h = (matrix + ridge_epsilon)**alpha
    error = 0
  else:
    damped_matrix = matrix + ridge_epsilon * identity

    z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
    new_mat_m_0 = damped_matrix * z
    new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
    _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(
        _iter_condition, _iter_body, init_state)
    error = jnp.max(jnp.abs(mat_m - identity))
    is_converged = jnp.asarray(convergence, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    resultant_mat_h = jnp.asarray(resultant_mat_h, matrix.dtype)
  return resultant_mat_h, error
