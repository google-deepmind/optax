# Lint as: python3
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
"""Distributed Shampoo Implementation."""

import enum
import functools
import itertools
from typing import Any, NamedTuple

import chex
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from optax._src import linear_algebra
from optax._src import transform
from optax._src import utils

# pylint:disable=no-value-for-parameter


class ParameterStats(NamedTuple):
  """State associated to each parameter of the model being trained."""
  diagonal_statistics: chex.Array  # Accumulator for diagonal preconditioner
  statistics: chex.Array    # Statistics
  preconditioners: chex.Array    # Preconditioners
  diagonal_momentum: chex.Array    # Momentum for the diagonal preconditioner
  momentum: chex.Array  # Momentum for the shampoo preconditioner


class ShampooState(transform.OptState):
  count: chex.Array
  stats: Any


class GraftingType(enum.Enum):
  SGD = 1
  ADAGRAD = 2


class BlockPartitioner:
  """Partitions a tensor into smaller tensors."""

  def __init__(self, param, block_size):
    self._shape = param.shape
    self._splits = []
    split_sizes = []
    # We split params into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(param.shape):
      if 0 < block_size < d:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d-1) // block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        split_sizes.append(sizes)
      else:
        split_sizes.append(np.array([d], dtype=np.int32))
    self._num_splits = len(split_sizes)
    self._preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])

  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return self._num_splits

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, indices) in self._splits:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
      tensors = tensors_local
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            jnp.concatenate(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, param, block_size, best_effort_shape_interpretation):
    self._original_shape = param.shape
    self._transformed_shape = param.shape
    if best_effort_shape_interpretation:
      self._transformed_shape = utils.merge_small_dims(
          self._original_shape, block_size)
    reshaped_param = jnp.reshape(param, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_param, block_size)

  def statistics_from_grad(self, grad):
    """Compute statistics from gradients.

    Args:
      grad: Gradient to compute statistics from.

    Returns:
      A list of gradient statistics for each partition.
    """
    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    stats = []
    for g in partitioned_grads:
      g_stats = []
      rank = len(g.shape)
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = jnp.tensordot(g, g, axes=(axes, axes))
        g_stats.append(stat)
      stats.extend(g_stats)
    return stats

  def shapes_for_preconditioners(self):
    """Returns shape from statistics."""
    return self._partitioner.shapes_for_preconditioners()

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    return 2 * len(self._transformed_shape)

  def preconditioned_grad(self, grad, preconditioners):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.
      preconditioners: A list of preconditioners to apply.

    Returns:
      A preconditioned gradient.
    """

    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, g in enumerate(partitioned_grads):
      preconditioners_for_grad = preconditioners[
          i * num_splits:(i + 1) * num_splits]
      rank = len(g.shape)
      precond_g = g
      for j in range(rank):
        precond_g = jnp.tensordot(
            precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
      preconditioned_partitioned_grads.append(precond_g)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return jnp.reshape(merged_grad, self._original_shape)


def distributed_shampoo(
    learning_rate,
    block_size,
    beta1=0.9,
    beta2=0.999,
    diagonal_epsilon=1e-10,
    matrix_epsilon=1e-6,
    weight_decay=0.0,
    start_preconditioning_step=1,
    preconditioning_compute_steps=1,
    statistics_compute_steps=1,
    best_effort_shape_interpretation=True,
    graft_type=GraftingType.SGD,
    nesterov=True,
    exponent_override=0,
    batch_axis_name=None,
    inverse_failure_threshold=0.1,
    precision=lax.Precision.HIGHEST):
  """Distributed Shampoo optimizer.

  Distributed Shampoo is a second-order preconditioned method (concretely, a
  variant of full-matrix Adagrad), that provides significant convergence and
  wall-clock time improvements compared to conventional first-order methods,
  and that has been shown to scale to large state-of-the-art deep learning
  models.

  References:
    [Anil et al.](https://arxiv.org/abs/2002.09018)

  Args:
    learning_rate: the step size used to update the parameters.
    block_size: Block size for large layers (if > 0). Preconditioning compute
      operation is cubic in the dimension of the tensor. Block size allows us
      to chunk the layers into sub-layers of maximal dimension dictated by
      this value. Use 128 as default (increase if you have compute budget).
    beta1: momentum parameter.
    beta2: second moment averaging parameter.
    diagonal_epsilon: epsilon for diagonal adagrad (only if layerwise grafting
      to AdaGrad is enabled).
    matrix_epsilon: epsilon to add to statistics before computing inverse pth
      root. If you are running in f32 precision for inverse pth root
      (recommended today) this can go upto 1e-6. If you have latest hardware
      with native f64 precision, set this upto 1e-12.
    weight_decay: Weight decay for regularization.
    start_preconditioning_step: When to start Shampoo update before which
      diagonal update is used. This is because we dont have enough information
      to do stable inverse.
    preconditioning_compute_steps: How often to compute preconditioner.
      Performance tuning params for controlling memory and compute
      requirements. Ideally set this and statistics_compute_steps params to 1.
    statistics_compute_steps: How often to compute statistics.
    best_effort_shape_interpretation: If there are some small dimensions,
      collapse them e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if
      block = 1024, [1, 2, 768, 1, 2048] --> [2, 768, 2048]
    graft_type: Grafting is a technique to fix the layerwise scale of Shampoo
      optimizer. This allows us to plugin the Shampoo optimizer into settings
      where SGD/AdaGrad is already well tuned. Available options are:
      GraftingType.SGD and GraftingType.ADAGRAD.
    nesterov: Nesterov momentum.
    exponent_override: Override the exponent used in matrix inverse.
    batch_axis_name: labeled axis over pmap for dataparallel training the
      optimizer used for.
    inverse_failure_threshold: numerics are hard and inverses fail
      sometimes; we determine that using this threshold.
    precision: precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise)
      b) lax.Precision.HIGH (increased precision, slower)
      c) lax.Precision.HIGHEST (best possible precision, slowest)

  Returns:
    a GradientTransformation.
  """

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      rank = len(param.shape)
      preconditioner = Preconditioner(
          param, block_size, best_effort_shape_interpretation)
      statistics = []
      preconditioners = []
      if rank >= 1:
        shapes = preconditioner.shapes_for_preconditioners()
        statistics = [matrix_epsilon * jnp.eye(s[0]) for s in shapes]
        preconditioners = [jnp.eye(s[0]) for s in shapes]

      adagrad_statistics = []
      if graft_type == GraftingType.ADAGRAD:
        adagrad_statistics = jnp.zeros_like(param)
      return ParameterStats(
          adagrad_statistics, statistics, preconditioners,
          jnp.zeros_like(param), jnp.zeros_like(param))

    return ShampooState(
        count=jnp.zeros([], jnp.int32),
        stats=jax.tree_map(_init, params))

  def _skip_preconditioning(param):
    return len(param.shape) < 1

  def _compute_stats(grad, state, param, step):
    """Compute per-parameter statistics."""
    preconditioner = Preconditioner(
        param, block_size, best_effort_shape_interpretation)
    new_statistics = [[]] * len(state.statistics)
    w1 = beta2
    w2 = beta2 if beta2 == 1.0 else (1.0 - beta2)
    if not _skip_preconditioning(param):
      def compute_updated_statistics():
        new_stats = preconditioner.statistics_from_grad(grad)
        new_stats_accumulators = []
        for stat, stat_accumulator in zip(new_stats, state.statistics):
          new_stats_accumulators.append(w1 * stat_accumulator + w2 * stat)
        return new_stats_accumulators

      if statistics_compute_steps > 1:
        perform_step = step % statistics_compute_steps == 0
        init_state = state.statistics
        new_statistics = list(utils.efficient_cond(
            perform_step, compute_updated_statistics, init_state))
      else:
        new_statistics = compute_updated_statistics()
    return ParameterStats(
        state.diagonal_statistics, new_statistics, state.preconditioners,
        state.diagonal_momentum, state.momentum)

  def _compute_preconditioners(states, step):
    """Compute preconditioners for statistics."""
    statistics = []
    num_statistics_per_state = []
    original_shapes = []
    exponents = []
    max_size = 0
    prev_preconditioners = []
    for state in states:
      num_statistics = len(state.statistics)
      num_statistics_per_state.append(num_statistics)
      original_shapes_for_state = []
      if num_statistics > 0:
        for statistic in state.statistics:
          exponents.append(
              2*num_statistics if exponent_override == 0 else exponent_override)
          original_shapes_for_state.append(statistic.shape)
          max_size = max(max_size, statistic.shape[0])
        statistics.extend(state.statistics)
        prev_preconditioners.extend(state.preconditioners)
        original_shapes.extend(original_shapes_for_state)
    num_statistics = len(statistics)

    if not batch_axis_name:
      num_devices = jax.local_device_count()
    else:
      num_devices = lax.psum(1, batch_axis_name)

    # Pad statistics and exponents to next multiple of num_devices.
    packed_statistics = [
        utils.pad_matrix(stat, max_size) for stat in statistics]
    to_pad = -num_statistics % num_devices
    packed_statistics.extend([
        jnp.eye(max_size, dtype=packed_statistics[0].dtype)
        for _ in range(to_pad)])
    exponents.extend([1 for _ in range(to_pad)])

    # Batch statistics and exponents so that so that leading axis is
    # num_devices.
    def _batch(statistics, exponents, num_devices):
      assert len(statistics) == len(exponents)
      n = len(statistics)
      b = int(n / num_devices)
      batched_statistics = [
          jnp.stack(statistics[idx:idx + b]) for idx in range(0, n, b)]
      batched_exponents = [
          jnp.stack(exponents[idx:idx + b]) for idx in range(0, n, b)]
      return jnp.stack(batched_statistics), jnp.stack(batched_exponents)

    # Unbatch values across leading axis and return a list of elements.
    def _unbatch(batched_values):
      b1, b2 = batched_values.shape[0], batched_values.shape[1]
      results = []
      for v_array in jnp.split(batched_values, indices_or_sections=b1, axis=0):
        v_array = jnp.squeeze(v_array)
        # b2 = batches (number of preconditioner computation) per core.
        if b2 > 1:
          for v in jnp.split(v_array, indices_or_sections=b2, axis=0):
            results.append(jnp.squeeze(v))
        else:
          results.append(v_array)

      return results

    all_statistics, all_exponents = _batch(
        packed_statistics, exponents, num_devices)

    def _matrix_inverse_pth_root(xs, ps):
      mi_pth_root = functools.partial(
          linear_algebra.matrix_inverse_pth_root,
          ridge_epsilon=matrix_epsilon, precision=precision)
      preconditioners, errors = jax.vmap(mi_pth_root)(xs, ps)
      return preconditioners, errors

    if not batch_axis_name:
      preconditioners, errors = jax.pmap(_matrix_inverse_pth_root)(
          all_statistics, all_exponents)
      preconditioners_flat = _unbatch(preconditioners)
      errors_flat = _unbatch(errors)
    else:

      def _internal_inverse_pth_root_all():
        preconditioners = jnp.array(all_statistics)
        current_replica = lax.axis_index(batch_axis_name)
        preconditioners, errors = _matrix_inverse_pth_root(
            all_statistics[current_replica], all_exponents[current_replica])
        preconditioners = jax.lax.all_gather(preconditioners,
                                             batch_axis_name)
        errors = jax.lax.all_gather(errors, batch_axis_name)
        preconditioners_flat = _unbatch(preconditioners)
        errors_flat = _unbatch(errors)
        return preconditioners_flat, errors_flat

      if preconditioning_compute_steps == 1:
        preconditioners_flat, errors_flat = _internal_inverse_pth_root_all()
      else:
        # Passing statistics instead of preconditioners as they are similarly
        # shaped tensors, as error we are passing is the threshold these will
        # be ignored.
        preconditioners_init = packed_statistics
        errors_init = ([inverse_failure_threshold] * len(packed_statistics))
        init_state = [preconditioners_init, errors_init]
        perform_step = step % preconditioning_compute_steps == 0
        preconditioners_flat, errors_flat = utils.efficient_cond(
            perform_step, _internal_inverse_pth_root_all, init_state)

    def _skip(error):
      condition = jnp.logical_or(
          jnp.isnan(error), error >= inverse_failure_threshold)
      return condition.astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
      return lax.cond(
          _skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_preconditioners_flat = []
    for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes,
                                       prev_preconditioners, errors_flat):
      new_preconditioners_flat.append(
          _select_preconditioner(error, p[:shape[0], :shape[1]], prev_p))

    assert len(states) == len(num_statistics_per_state)
    assert len(new_preconditioners_flat) == num_statistics

    # Add back empty preconditioners so we that we can set the optimizer state.
    preconditioners_for_states = []
    idx = 0
    for num_statistics, state in zip(num_statistics_per_state, states):
      if num_statistics == 0:
        preconditioners_for_states.append([])
      else:
        preconditioners_for_state = new_preconditioners_flat[idx:idx +
                                                             num_statistics]
        assert len(state.statistics) == len(preconditioners_for_state)
        preconditioners_for_states.append(preconditioners_for_state)
        idx += num_statistics
    new_states = []
    for state, new_preconditioners in zip(states, preconditioners_for_states):
      new_states.append(
          ParameterStats(
              state.diagonal_statistics, state.statistics, new_preconditioners,
              state.diagonal_momentum, state.momentum))

    return new_states

  def _transform_grad(grad, state, param, step):
    """Transform per-parameter gradients."""
    preconditioner = Preconditioner(
        param, block_size, best_effort_shape_interpretation)
    sgd_update = grad
    new_diagonal_statistics = state.diagonal_statistics
    if graft_type == GraftingType.ADAGRAD:
      new_diagonal_statistics = state.diagonal_statistics + jnp.square(grad)
      adagrad_update = grad / (
          jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)
      grafting_update = adagrad_update
    else:
      grafting_update = sgd_update

    precond_grad = grad
    if not _skip_preconditioning(param):
      precond_grad = preconditioner.preconditioned_grad(
          precond_grad, state.preconditioners)
    else:
      precond_grad = grafting_update

    grafting_update_norm = jnp.linalg.norm(grafting_update)
    precond_grad_norm = jnp.linalg.norm(precond_grad)
    shampoo_update = precond_grad * (
        grafting_update_norm / (precond_grad_norm + 1e-16))

    shampoo_update_with_wd = shampoo_update
    grafting_update_with_wd = grafting_update
    if weight_decay != 0:
      shampoo_update_with_wd = shampoo_update + weight_decay * param
      grafting_update_with_wd = grafting_update + weight_decay * param

    shampoo_update_with_wd_momentum = (
        state.momentum * beta1 + shampoo_update_with_wd)
    grafting_update_with_wd_momentum = (
        state.diagonal_momentum * beta1 + grafting_update_with_wd)

    run_shampoo = (step >= start_preconditioning_step).astype(
        grafting_update_with_wd_momentum.dtype)

    momentum_update = (
        run_shampoo * shampoo_update_with_wd_momentum +
        (1.0 - run_shampoo) * grafting_update_with_wd_momentum)

    wd_update = (
        run_shampoo * shampoo_update_with_wd +
        (1.0 - run_shampoo) * grafting_update_with_wd)

    if nesterov:
      momentum_update = wd_update + beta1 * momentum_update

    transformed_update = - learning_rate * momentum_update
    param_stats = ParameterStats(
        new_diagonal_statistics, state.statistics, state.preconditioners,
        grafting_update_with_wd_momentum, shampoo_update_with_wd_momentum)
    return transformed_update, param_stats

  def update_fn(grads, state, params):
    """Transform the input gradient and update all statistics.

    Args:
      grads: the gradient tensors for the parameters.
      state: a named tuple containing the state of the optimizer
      params: the parameters that should be updated.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    params_flat, treedef = jax.tree_flatten(params)
    stats_flat = treedef.flatten_up_to(state.stats)
    grads_flat = treedef.flatten_up_to(grads)

    new_stats_flat = jax.tree_multimap(
        lambda g, s, p: _compute_stats(g, s, p, state.count),
        grads_flat, stats_flat, params_flat)
    new_stats_flat = _compute_preconditioners(new_stats_flat, state.count)

    outputs = jax.tree_multimap(
        lambda g, s, p: _transform_grad(g, s, p, state.count),
        grads_flat, stats_flat, params_flat)
    updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

    updates = jax.tree_unflatten(treedef, updates_flat)
    new_stats = jax.tree_unflatten(treedef, new_stats_flat)

    new_state = ShampooState(
        count=utils.safe_int32_increment(state.count), stats=new_stats)
    return updates, new_state

  return transform.GradientTransformation(init_fn, update_fn)
