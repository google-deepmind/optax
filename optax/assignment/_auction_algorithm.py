# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""The Auction algorithm for the linear assignment problem.

Hybrid implementation aiming to be fast on both GPU and CPU:

* On GPU/TPU: uses a vectorized variant that processes all rows each
  iteration, but avoids building a dense [n_rows, n_cols] bid matrix. Column
  maxima and winners are computed via scatter-max reductions over 1D arrays,
  which is much lighter on memory and bandwidth.
* On CPU: uses a per-row Auction body with a custom best/second-best
  reduction, which avoids heavy primitives like top_k, making it more
  suitable for CPU backends.

`auction_algorithm(...)` automatically selects the implementation by default,
but users can override this with the `implementation` argument, which is
especially useful when calling `jax.jit(auction_algorithm, backend=...)`.
"""

from typing import Optional, Tuple
import jax
from jax import lax
import jax.numpy as jnp
from jax import core as jax_core


def _first_true(mask: jax.Array) -> jax.Array:
  """Return the index of the first True in a 1D boolean mask."""
  idx = jnp.arange(mask.shape[0], dtype=jnp.int32)
  masked = jnp.where(mask, idx, idx.shape[0])
  return jnp.argmin(masked).astype(jnp.int32)


def _best_two_max(values: jax.Array
                  ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Return (best_val, best_idx, second_val, second_idx) for a 1D array.

  This is a custom 1D reduction that finds the maximum and second maximum
  in a single pass using lax.fori_loop, which tends to be lighter than
  calling lax.top_k on CPU.
  """
  n = values.shape[0]
  dtype = values.dtype
  neg_inf = jnp.array(-jnp.inf, dtype=dtype)

  def body(i, carry):
    best_val, best_idx, second_val, second_idx = carry
    v = values[i]

    # If v is better than the current best, shift best -> second, v -> best.
    better_than_best = v > best_val
    new_best_val = jnp.where(better_than_best, v, best_val)
    new_best_idx = jnp.where(better_than_best, i, best_idx)

    # For second best: if we just updated best, old best becomes new second.
    candidate_second_val = jnp.where(better_than_best, best_val, second_val)
    candidate_second_idx = jnp.where(better_than_best, best_idx, second_idx)

    # Otherwise, check if v fits as second best.
    better_than_second = jnp.logical_and(~better_than_best,
                                         v > candidate_second_val)
    new_second_val = jnp.where(better_than_second, v, candidate_second_val)
    new_second_idx = jnp.where(better_than_second, i, candidate_second_idx)

    return (new_best_val, new_best_idx, new_second_val, new_second_idx)

  init = (
      neg_inf,
      jnp.array(-1, dtype=jnp.int32),
      neg_inf,
      jnp.array(-1, dtype=jnp.int32),
  )
  best_val, best_idx, second_val, second_idx = lax.fori_loop(0, n, body, init)
  return best_val, best_idx, second_val, second_idx


# Vectorized core: tuned for GPU/TPU
@jax.jit
def _auction_vectorized_core(
    cost_matrix: jax.Array,
    eps: jax.Array,
    max_iterations: int,
) -> Tuple[jax.Array, jax.Array]:
  """Vectorized Auction core for accelerators.

  Processes all rows in each iteration using dense tensor ops. To reduce
  memory traffic, it avoids constructing a dense [n_rows, n_cols] bid matrix
  and instead computes per-column maxima and winning rows with scatter-max
  reductions over 1D arrays.

  Args:
    cost_matrix: 2D array of shape (n_rows, n_cols), float32.
    eps: Scalar epsilon (float32).
    max_iterations: Integer maximum iterations.

  Returns:
    row2col: int32 array of shape (n_rows,).
    num_unassigned: scalar int32; > 0 if max_iterations was hit early.
  """
  n_rows, n_cols = cost_matrix.shape

  prices = jnp.zeros(n_cols, dtype=cost_matrix.dtype)
  row2col = jnp.full(n_rows, -1, dtype=jnp.int32)   # row -> column
  col2row = jnp.full(n_cols, -1, dtype=jnp.int32)   # column -> row
  num_unassigned = jnp.array(n_rows, dtype=jnp.int32)
  it0 = jnp.array(0, dtype=jnp.int32)

  row_indices = jnp.arange(n_rows, dtype=jnp.int32)

  def cond(state):
    prices, row2col, col2row, num_unassigned, it = state
    return jnp.logical_and(num_unassigned > 0, it < max_iterations)

  def body(state):
    prices, row2col, col2row, num_unassigned, it = state

    unassigned_mask = (row2col == -1)

    # utilities[i, j] = -cost[i, j] - price[j]
    utilities = -cost_matrix - prices

    vals, idxs = lax.top_k(utilities, k=2)
    best_val = vals[:, 0]
    second_val = vals[:, 1]
    best_col = idxs[:, 0].astype(jnp.int32)

    bid = best_val - second_val + eps
    bid = jnp.where(unassigned_mask, bid, 0.0)

    # Scatter-max over columns to get column_max and winning_row.
    column_max = jnp.zeros(n_cols, dtype=cost_matrix.dtype).at[best_col].max(bid)
    has_bid_mask = column_max > 0.0

    col_max_for_row = column_max[best_col]
    is_winner_row = (bid > 0.0) & (bid == col_max_for_row)

    winning_row = jnp.full(n_cols, -1, dtype=jnp.int32)
    winning_row = winning_row.at[best_col].max(
        jnp.where(is_winner_row, row_indices, -1)
    )

    prev_owner = col2row

    prices = prices + column_max * has_bid_mask.astype(cost_matrix.dtype)

    # Unassign previous owners.
    valid_prev = (prev_owner != -1) & has_bid_mask
    rows_prev = jnp.where(valid_prev, prev_owner, 0)
    weights_prev = valid_prev.astype(jnp.int32)

    row_unassign_count = jnp.zeros(n_rows, jnp.int32).at[rows_prev].add(
        weights_prev
    )
    row_unassign_mask = row_unassign_count > 0
    row2col = jnp.where(row_unassign_mask, -1, row2col)

    # Assign new owners.
    cols_valid = jnp.arange(n_cols, dtype=jnp.int32)
    cols_valid = jnp.where(has_bid_mask, cols_valid, 0)
    rows_valid = jnp.where(has_bid_mask, winning_row, 0)
    weights_assign = has_bid_mask.astype(jnp.int32)

    row_assign_count = jnp.zeros(n_rows, jnp.int32).at[rows_valid].add(
        weights_assign
    )
    row_assign_sum = jnp.zeros(n_rows, jnp.int32).at[rows_valid].add(cols_valid)

    new_assign_row = jnp.where(row_assign_count > 0, row_assign_sum, -1)

    row2col = jnp.where(new_assign_row != -1, new_assign_row, row2col)
    col2row = jnp.where(has_bid_mask, winning_row, col2row)

    num_unassigned = jnp.sum(row2col == -1).astype(jnp.int32)
    it = it + 1
    return prices, row2col, col2row, num_unassigned, it

  prices, row2col, col2row, num_unassigned, it = lax.while_loop(
      cond, body, (prices, row2col, col2row, num_unassigned, it0)
  )

  return row2col, num_unassigned


# Sequential core: tuned for CPU

@jax.jit
def _auction_single_row_core(
    cost_matrix: jax.Array,
    eps: jax.Array,
    max_iterations: int,
) -> Tuple[jax.Array, jax.Array]:
  """Single-row Auction core, optimized for CPU.

  In each iteration, picks ONE unassigned row (the smallest index), computes its
  utilities against all columns, and performs a single-row auction update.

  Uses `_best_two_max` instead of `lax.top_k` and avoids any dense bid matrix,
  which tends to be significantly lighter on CPU.

  Args:
    cost_matrix: 2D array of shape (n_rows, n_cols), float32.
    eps: Scalar epsilon (float32).
    max_iterations: Integer maximum iterations.

  Returns:
    row2col: int32 array of shape (n_rows,).
    num_unassigned: scalar int32; > 0 if max_iterations was hit early.
  """
  n_rows, n_cols = cost_matrix.shape

  prices = jnp.zeros(n_cols, dtype=cost_matrix.dtype)
  row2col = jnp.full(n_rows, -1, dtype=jnp.int32)
  col2row = jnp.full(n_cols, -1, dtype=jnp.int32)
  num_unassigned = jnp.array(n_rows, dtype=jnp.int32)
  it0 = jnp.array(0, dtype=jnp.int32)

  def cond(state):
    prices, row2col, col2row, num_unassigned, it = state
    return jnp.logical_and(num_unassigned > 0, it < max_iterations)

  def body(state):
    prices, row2col, col2row, num_unassigned, it = state

    unassigned_mask = (row2col == -1)
    row_idx = _first_true(unassigned_mask)

    utilities = -cost_matrix[row_idx, :] - prices

    best_val, best_col, second_val, _ = _best_two_max(utilities)
    best_col = best_col.astype(jnp.int32)

    bid = best_val - second_val + eps

    prev_owner = col2row[best_col]
    prices = prices.at[best_col].add(bid)

    row2col = row2col.at[row_idx].set(best_col)
    col2row = col2row.at[best_col].set(row_idx)

    def unassign_old(r2c):
      return r2c.at[prev_owner].set(-1)

    row2col = lax.cond(prev_owner != -1, unassign_old, lambda r2c: r2c, row2col)

    def dec(n):
      return n - 1

    num_unassigned = lax.cond(prev_owner == -1, dec, lambda n: n, num_unassigned)

    it = it + 1
    return prices, row2col, col2row, num_unassigned, it

  prices, row2col, col2row, num_unassigned, it = lax.while_loop(
      cond, body, (prices, row2col, col2row, num_unassigned, it0)
  )

  return row2col, num_unassigned


# Public API: choose core based on backend / implementation

def auction_algorithm(
    cost_matrix: jax.Array,
    epsilon: float = 1e-3,
    max_iterations: Optional[int] = None,
    implementation: str = "auto",
) -> Tuple[jax.Array, jax.Array]:
  r"""The Auction algorithm for the linear assignment problem.

  Given a cost matrix :math:`C \in \mathbb{R}^{n \times m}`, the goal is to
  select :math:`\min(n, m)` pairs of row/column indices such that:

  * each selected pair :math:`(i, j)` is unique in its row and column,
  * each row appears in at most one pair,
  * each column appears in at most one pair,
  * the sum of the selected costs :math:`\sum C_{ij}` is minimized.

  This implementation uses the forward Auction algorithm and works for
  rectangular matrices by internally transposing the matrix when the number of
  rows exceeds the number of columns.

  It selects between two JIT-compiled cores:

  * `"vectorized"`: `_auction_vectorized_core`, which is heavily vectorized
    over all rows and columns and uses scatter-max reductions on 1D arrays.
    This is ideal for accelerators (GPU/TPU).
  * `"single_row"`: `_auction_single_row_core`, which does a per-row Auction
    update with a custom best/second-best reduction. This is tuned for CPU.

  The `implementation` argument controls which core is used:

    * `"auto"` (default): chooses `"vectorized"` if the default backend is
      `"gpu"` or `"tpu"`, otherwise `"single_row"`.
    * `"vectorized"`: always use the vectorized core.
    * `"single_row"`: always use the single-row core.

  When wrapping this function in `jax.jit`, users who pass an explicit backend
  (e.g. `backend="cpu"`) are encouraged to also fix `implementation`, e.g.:

      jax.jit(lambda C: auction_algorithm(C, implementation="single_row"),
              backend="cpu")

  Post-check for max_iterations
  -----------------------------
  Internally, the cores also track `num_unassigned` (number of rows with no
  assignment). If `max_iterations` is reached with some rows still unassigned,
  then:

    * In **eager / non-JIT** mode, a `RuntimeError` is raised.
    * Under `jax.jit`, `num_unassigned` is a tracer, so the Python-side
      post-check is skipped. In that case, users can check the result by
      verifying that no entries in the returned column indices are negative.

  Args:
    cost_matrix: A 2D JAX array of shape ``(n, m)`` containing costs. It is
      converted to float32 internally.
    epsilon: Positive scalar that controls the bidding step. Smaller values
      give solutions closer to the exact optimum but may require more
      iterations.
    max_iterations: Optional upper bound on the number of auction iterations.
      If ``None``, a default of ``10 * n * m`` is used.
    implementation: One of ``"auto"``, ``"vectorized"``, or ``"single_row"``.

  Returns:
    A pair ``(i, j)`` where ``i`` and ``j`` are 1D int32 JAX arrays of the
    same length, containing row and column indices of the assignment. The
    total assignment cost is ``cost_matrix[i, j].sum()``.

  Raises (eager mode only):
    RuntimeError: If `max_iterations` is reached before all rows are assigned.
      In that case some entries of the column index array would be -1; we
      prefer to fail loudly rather than silently returning a partial
      assignment.
  """
  cost_matrix = jnp.asarray(cost_matrix, dtype=jnp.float32)
  if cost_matrix.ndim != 2:
    raise ValueError(f"cost_matrix must be 2D, got shape {cost_matrix.shape}")

  n_rows, n_cols = cost_matrix.shape

  if n_rows == 0 or n_cols == 0:
    return jnp.zeros(0, jnp.int32), jnp.zeros(0, jnp.int32)

  # Ensure we have at most as many rows ("persons") as columns ("objects").
  transpose = n_rows > n_cols
  if transpose:
    cost_matrix = cost_matrix.T
    n_rows, n_cols = cost_matrix.shape

  # Handle the trivial single-row case directly.
  if n_rows == 1:
    j = jnp.argmin(cost_matrix[0]).astype(jnp.int32)
    if transpose:
      return jnp.array([j], dtype=jnp.int32), jnp.array([0], dtype=jnp.int32)
    else:
      return jnp.array([0], dtype=jnp.int32), jnp.array([j], dtype=jnp.int32)

  if max_iterations is None:
    max_iterations = int(10 * n_rows * n_cols)

  eps = jnp.asarray(epsilon, dtype=cost_matrix.dtype)

  impl = implementation.lower()
  if impl == "auto":
    backend = jax.default_backend()
    use_vectorized = backend in ("gpu", "tpu")
  elif impl == "vectorized":
    use_vectorized = True
  elif impl == "single_row":
    use_vectorized = False
  else:
    raise ValueError(
        "implementation must be one of 'auto', 'vectorized', 'single_row', "
        f"got {implementation!r}"
    )

  if use_vectorized:
    row2col, num_unassigned = _auction_vectorized_core(cost_matrix, eps,
                                                       max_iterations)
  else:
    row2col, num_unassigned = _auction_single_row_core(cost_matrix, eps,
                                                       max_iterations)

  # Post-check: ensure that all rows are assigned.
  # NOTE: this check is only performed in eager / non-traced mode; if
  # `auction_algorithm` is wrapped in `jax.jit`, num_unassigned will be a
  # Tracer and this block will be skipped (see docstring).
  if not isinstance(num_unassigned, jax_core.Tracer):
    num_unassigned_py = int(num_unassigned)
    if num_unassigned_py != 0:
      raise RuntimeError(
          "Auction algorithm did not assign all rows before reaching "
          f"max_iterations={max_iterations}. "
          "Consider increasing `max_iterations` or using a larger `epsilon`."
      )

  # Build the final row/column index arrays.
  if transpose:
    cols = jnp.arange(row2col.shape[0], dtype=jnp.int32)
    rows = row2col
    return rows, cols
  else:
    rows = jnp.arange(row2col.shape[0], dtype=jnp.int32)
    cols = row2col
    return rows, cols
