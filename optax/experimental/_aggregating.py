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

from typing import Any, Protocol

import chex
import jax
from optax import tree
from optax._src import base


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
