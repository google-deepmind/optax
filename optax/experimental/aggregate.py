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
"""Gradient transformations that aggregate gradients."""

import functools
from typing import Any, NamedTuple, Protocol
import chex
import jax
import jax.numpy as jnp
from optax import tree
from optax._src import base
from optax._src import utils
from optax.transforms import _combining


###############################################################################
# Aggregators


PerElementUpdates = chex.ArrayTree
AggregatedUpdates = chex.ArrayTree
MaybeAxis = int | list[int] | None


class AggregatorUpdateFn(Protocol):
  """Update function for aggregators."""

  def __call__(
      self,
      per_elt_updates: PerElementUpdates,
      state: base.OptState,
      params: base.Params | None = None,
      **extra_args: Any,
  ) -> tuple[AggregatedUpdates, base.OptState]:
    """Transforms per-element updates into aggregated update direction."""


class Aggregator(base.GradientTransformationExtraArgs):
  """A pair of pure functions that implement stateful aggregation of gradients.

  This class differs from a standard optax GradientTransformation as it is
  defined to operate on a set of invidividual gradients, rather than on
  aggregated gradients -- like the mini-batch average of gradients.

  Optax base GradientTransformation expect input and output updates to be of the
  same shape as the parameters. The aggregators take as inputs per-example
  gradients of shape [*batch_shape, *params_shape] and return update direction
  of shape [*params_shape].

  While usual optax transformations are used in an api of the form
    grads = jax.grad(loss)(params, batch)
    updates, opt_state = transformation.update(grads, opt_state)
  The aggregators are used in an api of the form
    grads = jax.vmap(jax.grad(loss), in_axes=(None, 0))(params, batch)
    updates, opt_state = aggregator.update(grads, opt_state)

  The signatures of AggregatorUpdateFn and GradientTransformationUpdateFn are
  identical, but the distinction is necessary for the user to adapt the gradient
  oracles to such specific transformations.

  Attributes:
    init: Initialization function that takes params and returns state.
    update: Update function that takes per-example gradients, state and params
      (optionally) and returns updates and updated state.
  """

  init: base.TransformInitFn
  update: AggregatorUpdateFn


def chain(*transforms) -> base.GradientTransformationExtraArgs:
  """Combines transforms, returning an Aggregator if one is present."""
  opt = _combining.chain(*transforms)
  if any(isinstance(t, Aggregator) for t in transforms):
    return Aggregator(opt.init, opt.update)
  return opt


#################################################################################


def process(
    preprocessor: base.GradientTransformation,
    aggregator: base.GradientTransformation | Aggregator,
    postprocessor: base.GradientTransformation,
    aggregator_has_aux: bool = False,
):
  """Process gradients through a sequence of transformations.

  Args:
    preprocessor: A transformation that maps per-example gradients to
      per-example updates.
    aggregator: A transformation that aggregates per-example updates into a
      single update.
    postprocessor: A transformation that maps aggregated updates to the final
      updates.
    aggregator_has_aux: Whether the aggregator returns more than just the
      average updates.

  Returns:
    A :class:`optax.GradientTransformation`.
  """

  def init_fn(params) -> tuple[base.OptState, base.OptState, base.OptState]:
    preprocess_state = preprocessor.init(params)
    aggregate_state = aggregator.init(params)
    postprocess_state = postprocessor.init(params)
    return preprocess_state, aggregate_state, postprocess_state

  def update_fn(indiv_grads, states, params=None, **extra_args):
    preprocess_state, aggregate_state, postprocess_state = states

    indiv_updates, new_preprocess_state = preprocessor.update(
        indiv_grads, preprocess_state, params, **extra_args
    )

    aggregated, new_aggregate_state = aggregator.update(
        indiv_updates, aggregate_state, params, **extra_args
    )

    if aggregator_has_aux:
      avg_updates, agg_aux = aggregated
      extra_args = extra_args | agg_aux
    else:
      avg_updates = aggregated

    ready_to_post_process = tree.get(new_aggregate_state, 'ready', True)

    updates, new_postprocess_state = jax.lax.cond(
        ready_to_post_process,
        lambda g, s, p, kw: postprocessor.update(g, s, p, **kw),
        lambda g, s, *_: (tree.zeros_like(avg_updates), s),
        avg_updates,
        postprocess_state,
        params,
        extra_args,
    )
    return updates, (
        new_preprocess_state,
        new_aggregate_state,
        new_postprocess_state,
    )

  if isinstance(aggregator, Aggregator):
    return Aggregator(init_fn, update_fn)
  else:
    return base.GradientTransformationExtraArgs(init_fn, update_fn)


################################################################################
# Base aggregator/accumulator


def average_per_element_udpates(
    per_elt_axis: int | list[int] = 0
) -> Aggregator:
  """Average per-element updates."""

  def update_fn(per_elt_updates, state, params=None):
    del params
    avg_updates = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis), per_elt_updates
    )
    return avg_updates, state

  return Aggregator(base.init_empty_state, update_fn)


class AccumulateAvgUpdatesState(NamedTuple):
  """State for the average gradient accumulator."""

  micro_step: int
  ready: bool
  avg_grad: base.Updates


def accumulate_avg_udpates(
    num_microbatches: int,
) -> base.GradientTransformation:
  """Accumulate average gradients."""

  if num_microbatches < 1:
    raise ValueError('num_microbatches must be larger than or equal to than 0.')

  if num_microbatches == 1:
    # If there is only one microbatch, we don't need accumulation.
    # We return identity to save unnecessary state tracking.
    return base.identity()

  def init_fn(params):
    return AccumulateAvgUpdatesState(
        micro_step=0, ready=False, avg_grad=tree.zeros_like(params)
    )

  def update_fn(updates, state, params=None):
    del params
    new_micro_step = state.micro_step + 1
    new_avg_grad = jax.tree.map(
        lambda u, a: a + (u - a) / new_micro_step,
        updates,
        state.avg_grad,
    )
    ready_state = AccumulateAvgUpdatesState(
        micro_step=0, ready=True, avg_grad=tree.zeros_like(new_avg_grad)
    )
    not_ready_state = AccumulateAvgUpdatesState(
        micro_step=new_micro_step, ready=False, avg_grad=new_avg_grad
    )
    updates, new_state = tree.where(
        new_micro_step == num_microbatches,
        (new_avg_grad, ready_state),
        (tree.zeros_like(new_avg_grad), not_ready_state),
    )
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def average_incrementally_updates(
    per_elt_axis: MaybeAxis, num_microbatches: int
) -> Aggregator | base.GradientTransformation:
  """Average and accumulate per-element updates."""
  if per_elt_axis is None:
    return accumulate_avg_udpates(num_microbatches)
  else:
    return chain(
        average_per_element_udpates(per_elt_axis),
        accumulate_avg_udpates(num_microbatches),
    )


################################################################################
# Adding mean and variance gradient metrics


def get_batch_size_from_per_elt_updates(
    per_elt_updates: base.Updates, per_elt_axis: MaybeAxis
) -> int:
  """Get batch size from per-element updates."""

  def get_batch_size(u):
    if isinstance(per_elt_axis, int):
      return u.shape[per_elt_axis]
    else:
      return functools.reduce(
          lambda a, b: a * b, [u.shape[i] for i in per_elt_axis]
      )

  batch_sizes = jax.tree.map(get_batch_size, per_elt_updates)
  batch_sizes = jax.tree.leaves(batch_sizes)
  if not all(b == batch_sizes[0] for b in batch_sizes):
    raise ValueError(
        f'Per-element updates must have the same batch size. Got: {batch_sizes}'
    )
  return batch_sizes[0]


class PerElementMeanAndSumSqDiffGradsState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  micro_step: int
  ready: bool
  mean_grads: base.Updates
  sum_sq_diff_grads: base.Updates


def get_per_element_mean_and_sum_sq_diff_grads(
    per_elt_axis: int | list[int] = 0,
    num_microbatches: int = 1,
) -> Aggregator:
  """Collect per-element variance metrics."""

  if per_elt_axis is None:
    raise NotImplementedError(
        'Per-element mean and sum square diff need a per_elt_axis.'
    )

  def compute_avg_and_sum_sq_diff(
      per_elt_udpates: base.Updates,
      state: base.OptState,
      params: base.Params | None,
  ) -> tuple[base.Updates, base.Updates]:
    del params
    batch_size = get_batch_size_from_per_elt_updates(
        per_elt_udpates, per_elt_axis
    )
    mean_grads = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis, keepdims=True),
        per_elt_udpates,
    )
    sum_sq_diff_grads = jax.tree.map(
        lambda x, a: jnp.sum(jnp.square(x - a), axis=per_elt_axis),
        per_elt_udpates,
        mean_grads,
    )
    mean_grads = jax.tree.map(
        lambda x: x.squeeze(axis=per_elt_axis), mean_grads
    )
    return (
        mean_grads,
        {'sum_sq_diff_grads': sum_sq_diff_grads, 'sample_size': batch_size},
    ), state

  if num_microbatches == 1:
    return Aggregator(base.init_empty_state, compute_avg_and_sum_sq_diff)

  def init_fn(params):
    return PerElementMeanAndSumSqDiffGradsState(
        micro_step=0,
        ready=False,
        mean_grads=tree.zeros_like(params),
        sum_sq_diff_grads=tree.zeros_like(params),
    )

  def update_fn(per_elt_udpates, state, params=None):
    del params
    batch_size = get_batch_size_from_per_elt_updates(
        per_elt_udpates, per_elt_axis
    )
    new_micro_step = state.micro_step + 1

    # Compute batch averages.
    batch_mean_grads = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis, keepdims=True), per_elt_udpates
    )
    batch_sum_sq_diff_grads = jax.tree.map(
        lambda x, a: jnp.sum(jnp.square(x - a), axis=per_elt_axis),
        per_elt_udpates,
        batch_mean_grads,
    )
    batch_mean_grads = jax.tree.map(
        lambda x: x.squeeze(axis=per_elt_axis), batch_mean_grads
    )

    # Update accumulated averages.
    delta = jax.tree.map(lambda u, a: u - a, batch_mean_grads, state.mean_grads)
    new_mean_grads = jax.tree.map(
        lambda a, d: a + d / new_micro_step,
        state.mean_grads,
        delta,
    )
    size_factor = state.micro_step * batch_size / new_micro_step
    new_sum_sq_diff_grads = jax.tree.map(
        lambda a, s, d: a + s + d**2 * size_factor,
        state.sum_sq_diff_grads,
        batch_sum_sq_diff_grads,
        delta,
    )
    maybe_outputs = (
        new_mean_grads,
        {
            'sum_sq_diff_grads': new_sum_sq_diff_grads,
            'sample_size': batch_size * new_micro_step,
        },
    )

    # Output or not the accumulated averages.
    ready_state = PerElementMeanAndSumSqDiffGradsState(
        micro_step=0,
        ready=True,
        mean_grads=tree.zeros_like(new_mean_grads),
        sum_sq_diff_grads=tree.zeros_like(new_sum_sq_diff_grads),
    )
    not_ready_state = PerElementMeanAndSumSqDiffGradsState(
        micro_step=new_micro_step,
        ready=False,
        mean_grads=new_mean_grads,
        sum_sq_diff_grads=new_sum_sq_diff_grads,
    )
    updates, new_state = tree.where(
        new_micro_step == num_microbatches,
        (maybe_outputs, ready_state),
        (tree.zeros_like(maybe_outputs), not_ready_state),
    )
    return updates, new_state

  return Aggregator(init_fn, update_fn)


class PerElementMeanAndVarianceEMAState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  count: jax.Array
  ema_decay: jax.Array
  mean_grads_ema: base.Updates
  variance_grads_ema: base.Updates


def track_per_element_mean_and_variance_with_ema(
    ema_decay: float = 0.9,
) -> base.GradientTransformation:
  """Track variance metrics with an EMA over time."""

  def init_fn(params):
    return PerElementMeanAndVarianceEMAState(
        count=jnp.zeros([], jnp.int32),
        ema_decay=jnp.asarray(ema_decay),
        mean_grads_ema=tree.zeros_like(params),
        variance_grads_ema=tree.zeros_like(params),
    )

  def update_fn(updates, state, params=None, *, sum_sq_diff_grads, sample_size):
    del params
    mean_grads_ema = jax.tree.map(
        lambda x, y: (1.0 - ema_decay) * x + ema_decay * y,
        updates,
        state.mean_grads_ema,
    )
    variance_step = tree.scale(1 / (sample_size - 1), sum_sq_diff_grads)
    variance_grads_ema = jax.tree.map(
        lambda x, y: (1.0 - ema_decay) * x + ema_decay * y,
        variance_step,
        state.variance_grads_ema,
    )
    new_count = utils.safe_int32_increment(state.count)
    new_state = state._replace(
        count=new_count,
        mean_grads_ema=mean_grads_ema,
        variance_grads_ema=variance_grads_ema,
    )
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def get_unbiased_mean_and_variance_ema(
    state: base.OptState,
) -> tuple[base.Updates, base.Updates]:
  """Track unbiased mean and variance with an EMA over time."""
  per_elt_mean_and_variance_ema_state = tree.get(
      state, 'PerElementMeanAndVarianceEMAState', None
  )
  if per_elt_mean_and_variance_ema_state is None:
    raise ValueError(
        'State must have PerElementMeanAndVarianceEMAState to compute unbiased'
        ' mean and variance EMA.'
    )
  count = per_elt_mean_and_variance_ema_state.count
  ema_decay = per_elt_mean_and_variance_ema_state.ema_decay
  mean_grads_ema = per_elt_mean_and_variance_ema_state.mean_grads_ema
  variance_grads_ema = per_elt_mean_and_variance_ema_state.variance_grads_ema
  unbiased_mean_grads_ema = jax.tree.map(
      lambda x: x / (1 - ema_decay**count), mean_grads_ema
  )
  unbiased_variance_grads_ema = jax.tree.map(
      lambda x: x / (1 - ema_decay**count), variance_grads_ema
  )
  return unbiased_mean_grads_ema, unbiased_variance_grads_ema


def add_mean_variance_to_opt(
    opt: base.GradientTransformation,
    ema_decay: float = 0.9,
    per_elt_axis: MaybeAxis = 0,
    num_microbatches: int = 1,
):
  """Add mean and variance to an optimizer."""
  return process(
      preprocessor=base.identity(),
      aggregator=get_per_element_mean_and_sum_sq_diff_grads(
          per_elt_axis, num_microbatches
      ),
      postprocessor=chain(
          track_per_element_mean_and_variance_with_ema(ema_decay),
          opt,
      ),
      aggregator_has_aux=True,
  )
