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

import math
from typing import Any, NamedTuple, Protocol, Sequence

import chex
import jax
import jax.numpy as jnp
from optax import tree
from optax._src import base
from optax._src import utils
from optax.transforms import _combining


PerElementUpdates = chex.ArrayTree
AggregatedUpdates = chex.ArrayTree
MaybeAxis = int | Sequence[int] | None


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


def process(
    preprocessor: base.GradientTransformation,
    aggregator: base.GradientTransformation | Aggregator,
    postprocessor: base.GradientTransformation,
    aggregator_has_aux: bool = False,
) -> base.GradientTransformation | Aggregator:
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

  Examples:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax.random as jrd
    >>> import optax
    >>> from optax.experimental import aggregating
    >>> clip_per_sample = jax.vmap(optax.projections.projection_l2_ball, 0)
    >>> average_grads = lambda gs: jax.tree.map(lambda g: jnp.mean(g, 0), gs)
    >>> opt = aggregating.process(
    ...     # clip per sample
    ...     optax.stateless(lambda gs, _: clip_per_sample(gs)),
    ...     # average
    ...     aggregating.Aggregator(
    ...         init=optax.init_empty_state,
    ...         update=lambda gs, s, _: (average_grads(gs), s)
    ...     ),
    ...     # usual optimizer there
    ...     optax.adam(learning_rate=1e-1),
    ... )
    >>> fun = lambda w, x, y: jnp.sum((x.dot(w)-y)**2)
    >>> n, k, d = 8, 2, 4
    >>> xs = jrd.normal(jrd.key(0), (n, d))
    >>> ys = jrd.normal(jrd.key(1), (n, k))
    >>> params = jrd.normal(jrd.key(2), (d, k))
    >>> state = opt.init(params)
    >>> value_and_grads = jax.vmap(jax.value_and_grad(fun), (None, 0, 0))
    >>> for i in range(3):
    ...   losses, grads = value_and_grads(params, xs, ys)
    ...   updates, state = opt.update(grads, state)
    ...   params = optax.apply_updates(params, updates)
    ...   print(f'Step: {i} | Batch loss: {jnp.mean(losses):.2e}')
    Step: 0 | Batch loss: 7.47e+00
    Step: 1 | Batch loss: 6.20e+00
    Step: 2 | Batch loss: 5.08e+00
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


def average_per_element_updates(
    per_elt_axis: int | Sequence[int] = 0,
) -> Aggregator:
  """Average per-element updates.

  Args:
    per_elt_axis: The axis to average over.

  Returns:
    An Aggregator that averages per-element updates.
  """

  def update_fn(per_elt_updates, state, params=None):
    del params
    avg_updates = jax.tree.map(
        lambda x: jnp.mean(x, axis=per_elt_axis), per_elt_updates
    )
    return avg_updates, state

  return Aggregator(base.init_empty_state, update_fn)


class AccumulateAvgUpdatesState(NamedTuple):
  """State for the average gradient accumulator."""

  micro_step: jax.Array  # int
  ready: jax.Array  # bool
  avg_grad: base.Updates


def accumulate_avg_updates(
    accumulation_steps: int,
) -> base.GradientTransformation:
  """Accumulates average gradients for `accumulation_steps` microbatches.

  Best used in combination with :func:`optax.experimental.process` to define
  an optimizer accumulating gradients over multiple microbatches, see example
  below.

  Args:
    accumulation_steps: The number of microbatches to accumulate over.

  Returns:
    An optax GradientTransformation that accumulates average gradients for
    `accumulation_steps` microbatches.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax.random as jrd
    >>> import optax
    >>> num_microbatches = 3
    >>> size_microbatch = 4
    >>> output_dim = 2
    >>> input_dim = 6
    >>> xs = jrd.normal(
    ...     jrd.key(0), (num_microbatches, size_microbatch, input_dim)
    ... )
    >>> ys = jrd.normal(
    ...     jrd.key(1), (num_microbatches, size_microbatch, output_dim)
    ... )
    >>> params = jrd.normal(jrd.key(2), (input_dim, output_dim))
    >>> # The following is equivalent to an effective batch size of
    >>> # accumulation_steps * size_microbatch
    >>> opt = optax.adam(learning_rate=0.01)
    >>> opt = process(
    ...     preprocessor=base.identity(),
    ...     aggregator=average_incrementally_updates(
    ...          per_elt_axis=None,
    ...          accumulation_steps=2,
    ...     ),
    ...     postprocessor=opt,
    ... )
    >>> fun = lambda w, x, y: jnp.mean(jnp.sum((x.dot(w)-y)**2, axis=-1))
    >>> state = opt.init(params)
    >>> for i, (x, y) in enumerate(zip(xs, ys)):
    ...   full_loss = fun(params, xs, ys)
    ...   loss, grads = jax.value_and_grad(fun)(params, x, y)
    ...   updates, state = opt.update(grads, state)
    ...   params = optax.apply_updates(params, updates)
    ...   print(f'Step: {i}|Batch loss: {loss:.2e}|Full loss: {full_loss:.2e}')
    Step: 0|Batch loss: 2.51e+01|Full loss: 1.49e+01
    Step: 1|Batch loss: 1.16e+01|Full loss: 1.49e+01
    Step: 2|Batch loss: 7.93e+00|Full loss: 1.46e+01
  """

  if accumulation_steps < 1:
    raise ValueError('accumulation_steps must be larger than or equal to 1.')

  if accumulation_steps == 1:
    # If there is only one microbatch, we don't need accumulation.
    # We return identity to save unnecessary state tracking.
    return base.identity()

  def init_fn(params):
    return AccumulateAvgUpdatesState(
        micro_step=jnp.asarray(0),
        ready=jnp.asarray(False),
        avg_grad=tree.zeros_like(params),
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
        micro_step=jnp.asarray(0),
        ready=jnp.asarray(True),
        avg_grad=tree.zeros_like(new_avg_grad),
    )
    not_ready_state = AccumulateAvgUpdatesState(
        micro_step=new_micro_step,
        ready=jnp.asarray(False),
        avg_grad=new_avg_grad,
    )
    updates, new_state = tree.where(
        new_micro_step == accumulation_steps,
        (new_avg_grad, ready_state),
        (tree.zeros_like(new_avg_grad), not_ready_state),
    )
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def average_incrementally_updates(
    per_elt_axis: MaybeAxis, accumulation_steps: int
) -> Aggregator | base.GradientTransformation:
  """Average and accumulate per-element updates.

  Args:
    per_elt_axis: The axis to average over, or None if no averaging is desired.
    accumulation_steps: The number of microbatches to accumulate over.

  Returns:
    An optax GradientTransformation or an Aggregator that averages and/or
    accumulates per-element updates.
  """
  if per_elt_axis is None:
    return accumulate_avg_updates(accumulation_steps)
  else:
    agg = _combining.chain(
        average_per_element_updates(per_elt_axis),
        accumulate_avg_updates(accumulation_steps),
    )
    return Aggregator(agg.init, agg.update)


################################################################################
# Adding mean and variance gradient metrics


def get_batch_size_from_per_elt_updates(
    per_elt_updates: base.Updates, per_elt_axis: int | list[int]
) -> int:
  """Get batch size from per-element updates.

  Args:
    per_elt_updates: The per-element updates.
    per_elt_axis: The axis to average over.

  Returns:
    The batch size.
  """

  def get_batch_size(u):
    if isinstance(per_elt_axis, int):
      return u.shape[per_elt_axis]
    else:
      return math.prod(u.shape[i] for i in per_elt_axis)

  batch_sizes = jax.tree.map(get_batch_size, per_elt_updates)
  batch_sizes = jax.tree.leaves(batch_sizes)
  if not all(b == batch_sizes[0] for b in batch_sizes):
    raise ValueError(
        f'Per-element updates must have the same batch size. Got: {batch_sizes}'
    )
  return batch_sizes[0]


class PerElementMeanAndSumSqDiffGradsState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  micro_step: jax.Array  # int
  ready: jax.Array  # bool
  mean_grads: base.Updates
  sum_sq_diff_grads: base.Updates


def get_per_element_mean_and_sum_sq_diff_grads(
    per_elt_axis: int | Sequence[int] = 0,
    accumulation_steps: int = 1,
) -> Aggregator:
  """Collect per-element mean and sum square diff gradients.

  See :func:`optax.experimental.aggregating.add_mean_variance_to_opt` for an
  example on how to use this function.

  Args:
    per_elt_axis: The axis to average over.
    accumulation_steps: The number of microbatches to accumulate over.

  Returns:
    An Aggregator that collects per-element mean and sum square diff gradients.
  """

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
        lambda x, a: jnp.sum((x - a)**2, axis=per_elt_axis),
        per_elt_udpates,
        mean_grads,
    )
    mean_grads = jax.tree.map(
        lambda x: x.squeeze(axis=per_elt_axis), mean_grads
    )
    aux_data = {
        'sum_sq_diff_grads': sum_sq_diff_grads,
        'sample_size': batch_size,
    }
    return (mean_grads, aux_data), state

  if accumulation_steps == 1:
    return Aggregator(base.init_empty_state, compute_avg_and_sum_sq_diff)

  def init_fn(params):
    return PerElementMeanAndSumSqDiffGradsState(
        micro_step=jnp.asarray(0, dtype=jnp.int32),
        ready=jnp.asarray(False),
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
        micro_step=jnp.asarray(0, dtype=jnp.int32),
        ready=jnp.asarray(True),
        mean_grads=tree.zeros_like(new_mean_grads),
        sum_sq_diff_grads=tree.zeros_like(new_sum_sq_diff_grads),
    )
    not_ready_state = PerElementMeanAndSumSqDiffGradsState(
        micro_step=new_micro_step,
        ready=jnp.asarray(False),
        mean_grads=new_mean_grads,
        sum_sq_diff_grads=new_sum_sq_diff_grads,
    )
    updates, new_state = tree.where(
        new_micro_step == accumulation_steps,
        (maybe_outputs, ready_state),
        (tree.zeros_like(maybe_outputs), not_ready_state),
    )
    return updates, new_state

  return Aggregator(init_fn, update_fn)


class PerElementMeanAndVarianceEMAState(NamedTuple):
  """State for the per-element mean and variance accumulator."""

  count: jax.Array  # int
  ema_decay: jax.Array  # float
  mean_grads_ema: base.Updates
  variance_grads_ema: base.Updates


def track_per_element_mean_and_variance_with_ema(
    ema_decay: jax.typing.ArrayLike = 0.9,
) -> base.GradientTransformation:
  """Track variance metrics with an EMA over time.

  See :func:`optax.experimental.aggregating.add_mean_variance_to_opt` for a
  complete example.

  Args:
    ema_decay: The EMA decay factor.

  Returns:
    A GradientTransformation that tracks per-element mean and variance with an
    EMA over time. The mean and variance are computed thanks to the auxiliary
    arguments provided by the `get_per_element_mean_and_sum_sq_diff_grads`
    aggregator when this transformation is linked with it via
    :func:`optax.experimental.aggregating.process`.
  """

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
  """Retrieve unbiased mean and variance from the state.

  Args:
    state: The state of the optimizer.

  Returns:
    A tuple of unbiased mean and variance EMA.
  """
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
    ema_decay: jax.typing.ArrayLike = 0.9,
    per_elt_axis: int | list[int] = 0,
    accumulation_steps: int = 1,
):
  r"""Add mean and variance to an optimizer.

  Args:
    opt: The optimizer to add mean and variance to.
    ema_decay: The EMA decay factor.
    per_elt_axis: The axis to average over.
    accumulation_steps: The number of microbatches to accumulate over.

  Returns:
    An optax GradientTransformation that adds mean and variance to an optimizer.

  Example:
  >>> import jax
  >>> import jax.numpy as jnp
  >>> import jax.random as jrd
  >>> import optax
  >>> from optax.experimental import aggregating
  >>> num_microbatches = 3
  >>> size_microbatch = 4
  >>> output_dim = 2
  >>> input_dim = 2
  >>> xs = jrd.normal(
  ...     jrd.key(0), (num_microbatches, size_microbatch, input_dim)
  ... )
  >>> ys = jrd.normal(
  ...     jrd.key(1), (num_microbatches, size_microbatch, output_dim)
  ... )
  >>> params = jrd.normal(jrd.key(2), (input_dim, output_dim))
  >>> opt = optax.adam(learning_rate=0.01)
  >>> opt = aggregating.add_mean_variance_to_opt(
  ...     opt=opt,
  ...     ema_decay=0.9,
  ...     per_elt_axis=0,
  ...     num_microbatches=1,
  ... )
  >>> fun = lambda w, x, y: jnp.mean(jnp.sum((x.dot(w)-y)**2, axis=-1))
  >>> values_and_grads = jax.vmap(jax.value_and_grad(fun), (None, 0, 0))
  >>> state = opt.init(params)
  >>> for i, (x, y) in enumerate(zip(xs, ys)):
  ...   full_loss = fun(params, xs, ys)
  ...   losses, grads = values_and_grads(params, x, y)
  ...   updates, state = opt.update(grads, state)
  ...   params = optax.apply_updates(params, updates)
  ...   mean_ema, var_ema = aggregating.get_unbiased_mean_and_variance_ema(
  ...       state
  ...   )
  ...   print(f'Step: {i}|Batch loss: {jnp.mean(losses):.2e}')
  ...   print(f'Mean EMA:\n {mean_ema}\nVariance EMA:\n {var_ema}')
  Step: 0|Batch loss: 7.46e+00
  Mean EMA:
  [[ 2.8991693e-04  3.8409345e+00]
  [-1.1956869e+00  4.7318892e+00]]
  Variance EMA:
  [[ 4.3785257 48.656933 ]
  [ 3.8709724 77.66371  ]]
  Step: 1|Batch loss: 5.52e+00
  Mean EMA:
  [[ 1.2733988  3.0736945]
  [-3.0100226  3.1490781]]
  Variance EMA:
  [[29.188576 30.26823 ]
  [13.972735 49.39477 ]]
  Step: 2|Batch loss: 5.15e+00
  Mean EMA:
  [[ 1.244118   2.2072985]
  [-2.719866   3.0309222]]
  Variance EMA:
  [[23.888937  27.574411 ]
  [12.0260725 32.332287 ]]
  """
  return process(
      preprocessor=base.identity(),
      aggregator=get_per_element_mean_and_sum_sq_diff_grads(
          per_elt_axis, accumulation_steps
      ),
      postprocessor=_combining.chain(
          track_per_element_mean_and_variance_with_ema(ema_decay),
          opt,
      ),
      aggregator_has_aux=True,
  )
