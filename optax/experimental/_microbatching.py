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

"""Module providing a general `microbatch` transformation."""

from __future__ import annotations

import dataclasses
import enum
import functools
from typing import Any, Callable, Sequence, TypeAlias

import chex
import jax
import jax.numpy as jnp


AccumulatorTree: TypeAlias = Any


@dataclasses.dataclass(frozen=True)
class Accumulator:
  """A class for accumulating values in a microbatched function.

  Given a list of microbatch function evaluations [x_0, ..., x_{n-1}], this
  object represents the program.

  ```
  carry = init(x_0)
  for i in range(1, n):
    carry = update(carry, x_i, i)
  return finalize(carry)
  ```

  Attributes:
    init: A function f(value, num_microbatches) that initializes the microbatch
      state from the function evaluation of the fist microbatch.
    update: A function f(carry, value, index, num_microbatches) that updates the
      microbatch state with the function evaluation of the current microbatch.
    finalize: A function f(carry, num_microbatches) that returns the final
      result from the final state.
    aggregate: A function f(per_microbatch_value) that aggregates
      per-microbatch values into a single value.
  """

  init: Callable[[chex.ArrayTree], chex.ArrayTree]
  update: Callable[[chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree]
  finalize: Callable[[chex.ArrayTree], chex.ArrayTree]
  aggregate: Callable[[chex.ArrayTree], chex.ArrayTree]


# pylint: disable=g-bare-generic
def _with_floating_check(fn: Callable) -> Callable:
  def wrapper(*args, **kwargs):
    dtypes, _ = jax.tree.flatten(jax.tree.map(jnp.dtype, (args, kwargs)))
    if not all(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes):
      raise ValueError(
          'MEAN and RUNNING_MEAN Accumulators require floating-point values.'
      )
    return fn(*args, **kwargs)
  return wrapper


def _identity(value: Any) -> Any:
  return value


def reshape_batch_axis(pytree: Any, microbatch_size: int):
  """Reshape pytree leaves to shape (num_microbatches, microbatch_size, ...)."""
  # If data is sharded along the 0th axis, using column-major order is important
  # to ensure that each microbatch is sharded in the same manner.
  # For example, if the data was sharded across 2 devices, each device would
  # handle one of the examples in each microbatch.
  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] --> [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

  return jax.tree.map(
      lambda x: x.reshape(-1, microbatch_size, *x.shape[1:], order='F'),
      pytree,
  )


def _lift(accumulator: Accumulator) -> Accumulator:
  """Lifts an array-based Accumulator to a PyTree-based Accumulator."""
  return Accumulator(
      lambda value: jax.tree.map(accumulator.init, value),
      lambda carry, value, i: jax.tree.map(
          lambda c, v: accumulator.update(c, v, i), carry, value
      ),
      lambda carry: jax.tree.map(accumulator.finalize, carry),
      lambda values: jax.tree.map(accumulator.aggregate, values),
  )


def _compose(accumulators: AccumulatorTree) -> Accumulator:
  """Composes a PyTree of Accumulators into a single Accumulator."""

  def init(values):
    return jax.tree.map(
        lambda acc, val: acc.init(val),
        accumulators,
        values,
    )

  def update(carry, value, index):
    return jax.tree.map(
        lambda acc, car, val: acc.update(car, val, index),
        accumulators,
        carry,
        value,
    )

  def finalize(carry):
    return jax.tree.map(
        lambda acc, car: acc.finalize(car),
        accumulators,
        carry,
    )

  def aggregate(values):
    return jax.tree.map(
        lambda acc, val: acc.accumulate(val), accumulators, values
    )

  return Accumulator(init, update, finalize, aggregate)


def _sum() -> Accumulator:
  return _lift(
      Accumulator(
          init=_identity,
          update=lambda carry, value, _: carry + value,
          finalize=_identity,
          aggregate=functools.partial(jnp.sum, axis=0),
      )
  )


def _mean(num_microbatches: int) -> Accumulator:
  return _lift(
      Accumulator(
          init=_with_floating_check(_identity),
          update=lambda carry, value, _: carry + value,
          finalize=lambda carry: carry / num_microbatches,
          aggregate=functools.partial(jnp.mean, axis=0),
      )
  )


def _running_mean() -> Accumulator:
  def update(carry, value, index):
    p = index / (index + 1)
    new_state = carry * p + value * (1 - p)
    return new_state

  return _lift(
      Accumulator(
          init=_with_floating_check(_identity),
          update=update,
          finalize=_identity,
          aggregate=functools.partial(jnp.mean, axis=0),
      )
  )


def _concat(num_microbatches: int) -> Accumulator:
  def init(value):
    return jnp.broadcast_to(value, (num_microbatches,) + value.shape)

  def update(carry, value, index):
    return carry.at[index].set(value)

  def finalize(carry):
    return carry.reshape(-1, *carry.shape[2:], order='F')

  return _lift(Accumulator(init, update, finalize, _identity))


class AccumulationType(enum.Enum):
  """The type of accumulation to perform."""
  MEAN = enum.auto()
  SUM = enum.auto()
  RUNNING_MEAN = enum.auto()
  CONCAT = enum.auto()


# In order to construct some accumulators (MEAN, CONCAT), we need to know the
# number of microbatches. But we don't want to force the user to specify that in
# advance, so we offer an enum-based API for specifying accumulation strategies.
def _canonicalize(
    tree: Accumulator | AccumulationType | AccumulatorTree,
    num_microbatches: int
) -> Accumulator:
  """Canonicalizes a PyTree of Accumulators/AccumulationTypes."""
  def fun(acc):
    if isinstance(acc, Accumulator):
      return acc
    match acc:
      case AccumulationType.MEAN:
        return _mean(num_microbatches)
      case AccumulationType.SUM:
        return _sum()
      case AccumulationType.RUNNING_MEAN:
        return _running_mean()
      case AccumulationType.CONCAT:
        return _concat(num_microbatches)
    raise ValueError(f'Unknown accumulator: {acc}')

  return _compose(jax.tree.map(fun, tree))


_DEFAULT = AccumulationType.SUM


def microbatch(
    fun: Callable[..., Any],
    argnums: int | Sequence[int],
    microbatch_size: int | None,
    accumulator: Accumulator | AccumulationType | AccumulatorTree = _DEFAULT,
    num_real_microbatches: int | None = None,
) -> Callable[..., Any]:
  """A general microbatching transformation.

  Conceptually, given ``fun``, this function returns a new function that does
  something like the following (for the case of SUM accumulator):

  .. code-block:: python

    def microbatched_fun(full_batch):
      accumulator = 0
      for microbatch in full_batch:
        accumulator += fun(microbatch)
      return accumulator

  where under the hood the ``for`` is implemented via a ``lax.fori_loop`` and
  hence forced to be sequential.

  This function is useful when evaluating ``fun`` on the full input batch
  exceeds available device memory. By splitting the batch into smaller
  microbatches and processing them sequentially, peak memory usage can be
  significantly reduced. Because the function is evaluated on smaller batches,
  this transformation requires knowledge of how the individual microbatch
  results should be combined back together (SUM, MEAN, or CONCAT). See the
  accumulator argument for more details.

  Example Usage:
    >>> import jax.numpy as jnp
    >>> from optax.experimental import microbatching
    >>> fun = lambda x: (x+1, jnp.sum(3*x))
    >>> data = jnp.array([1, 2, 3, 4])
    >>> fun(data)
    (Array([2, 3, 4, 5], dtype=int32), Array(30, dtype=int32))
    >>> strategy = (
    ...    microbatching.AccumulationType.CONCAT,
    ...    microbatching.AccumulationType.SUM
    ... )
    >>> microbatched_fun = microbatching.microbatch(
    ...    fun, argnums=0, microbatch_size=2, accumulator=strategy
    ... )
    >>> microbatched_fun(data)
    (Array([2, 3, 4, 5], dtype=int32), Array(30, dtype=int32))

  Args:
      fun: An arbitrary function. All kwargs are assumed to have a batch axis.
      argnums: A sequence of argument indices that have a batch axis. All
        kwargs are assumed to have a batch axis, similar to ``jax.vmap``.
      microbatch_size: The number of rows in the overall batch used in each
        microbatch. Smaller values reduce memory overhead, but require more
        sequential computation. This must evenly divide the batch axis size of
        the batch arguments.
      accumulator: Specifies how to combine results from each microbatch; can be
        a single ``Accumulator``, a pytree matching the structure of ``fun``'s
        output, with ``Accumulator`` values at the leaves, or anything in
        between (i.e., a PyTree prefix of ``fun``'s output`).
      num_real_microbatches: Optional number of microbatches that are actually
        executed. If specified, microbatching will terminate early after this
        many steps. Can be helpful to handle variable batch sizes without
        recompilation.

  Returns:
      A new function that evaluates fun sequentially num_microbatches times on
        subsets of data. Consumes the same args and kwargs as ``fun``.
  """
  if microbatch_size is None:
    return fun

  if isinstance(argnums, int):
    argnums = (argnums,)

  def microbatched_fun(*args, **kwargs):
    batch_args = [args[i] for i in argnums]
    batch_size = jax.tree.leaves(batch_args)[0].shape[0]
    if batch_size % microbatch_size != 0:
      raise ValueError(f'{batch_size=} not divisible by {microbatch_size=}')
    num_microbatches = batch_size // microbatch_size
    accumulator_ = _canonicalize(accumulator, num_microbatches)

    reshaped_batch_args = reshape_batch_axis(batch_args, microbatch_size)
    reshaped_kwargs = reshape_batch_axis(kwargs, microbatch_size)

    def f(index):
      fetch = lambda arg: jax.tree.map(lambda x: x[index], arg)
      inputs = list(args)
      for i, arg in zip(argnums, reshaped_batch_args):
        inputs[i] = fetch(arg)
      input_kwargs = {k: fetch(kwarg) for k, kwarg in reshaped_kwargs.items()}
      return fun(*inputs, **input_kwargs)

    def body_fun(index, carry):
      return accumulator_.update(carry, f(index), index)

    loop_bound = num_real_microbatches or num_microbatches
    answer = jax.lax.fori_loop(
        1, loop_bound, body_fun, accumulator_.init(f(0))
    )

    return accumulator_.finalize(answer)

  return microbatched_fun
