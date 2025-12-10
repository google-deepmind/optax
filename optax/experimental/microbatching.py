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

import collections
import dataclasses
import enum
import functools
from typing import Any, Callable, Sequence, TypeAlias

import chex
import jax
import jax.numpy as jnp


AccumulatorTree: TypeAlias = Any
Function: TypeAlias = Callable[..., Any]
VmapFn: TypeAlias = Callable[[Function, int | Sequence[int], int], Function]
PyTreeFn: TypeAlias = Callable[[chex.ArrayTree], chex.ArrayTree]
UpdateFn = Callable[[chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree]
IndividualOutputs = collections.namedtuple('Aux', ['values', 'metrics', 'aux'])
ValueAndGradFn: TypeAlias = Callable[..., tuple[Any, IndividualOutputs]]


@dataclasses.dataclass(frozen=True)
class Accumulator:
  """A class for accumulating values in a microbatched function.

  Given a list of microbatch function evaluations [x_0, ..., x_{n-1}], this
  object represents the program.

  .. code-block:: python

  carry = init(x_0)
  for i in range(1, n):
    carry = update(carry, x_i, i)
  return finalize(carry)

  Attributes:
    init: A function f(value) that initializes the microbatch state from the
      function evaluation of the fist microbatch.
    update: A function f(carry, value, index) that updates the microbatch state
      with the function evaluation of the current microbatch.
    finalize: A function f(carry) that returns the final result from the final
      state.
    aggregate: A function f(per_microbatch_value) that aggregates
      per-microbatch values into a single value. Used by `micro_vmap`.
  """

  init: PyTreeFn
  update: UpdateFn
  finalize: PyTreeFn
  aggregate: PyTreeFn


def _with_floating_check(fn: Function) -> Function:
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


def reshape_batch_axis(tree: Any, microbatch_size: int, axis: int = 0) -> Any:
  """Reshape batch axis of pytree leaves for use with microbatching.

  This function reshapes the batch axis of each leaf into a shape
  (num_microbatches, microbatch_size) appearing at the same axis as the original
  batch axis. The reshape is done using a column-major order, so any sharding
  along the batch axis should be preserved in the new `microbatch_size` axis,
  while the new `num_microbatches` axis will generally be replicated.

  Args:
    tree: A pytree of jax.Arrays, each having a batch axis.
    microbatch_size: The size of sub-batches used for each microbatch.
    axis: The axis to reshape.

  Returns:
    A pytree of reshaped jax.Arrays.
  """
  def reshape_leaf(x):
    new_shape = x.shape[:axis] + (-1, microbatch_size) + x.shape[axis+1:]
    if jax.__version__ < '0.7.0':
      return x.reshape(new_shape, order='F')

    sharding = jax.typeof(x).sharding
    if not sharding.mesh.are_all_axes_explicit:
      return x.reshape(new_shape, order='F')

    assert jax.__version__ >= '0.8.1', (
        'microbatching with explicit sharding requires jax version >= 0.8.1.'
    )
    spec = sharding.spec
    if len(spec) < axis:  # The batch axis is not sharded.
      new_spec = spec
    else:
      new_spec = jax.P(*spec[:axis], None, spec[axis], *spec[axis+1:])
    out_sharding = jax.sharding.NamedSharding(sharding.mesh, new_spec)

    local_shape = sharding.shard_shape(x.shape)
    nshards = x.shape[axis] // local_shape[axis]
    if microbatch_size % nshards != 0:
      raise ValueError(f'{nshards=} must evenly divide {microbatch_size=}.')

    return x.reshape(new_shape, order='F', out_sharding=out_sharding)

  return jax.tree.map(reshape_leaf, tree)


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
        lambda acc, val: acc.aggregate(val), accumulators, values
    )

  return Accumulator(init, update, finalize, aggregate)


def _sum() -> Accumulator:
  """An Accumulator that computes the sum of microbatched outputs."""
  return _lift(
      Accumulator(
          init=_identity,
          update=lambda carry, value, _: carry + value,
          finalize=_identity,
          aggregate=functools.partial(jnp.sum, axis=0),
      )
  )


def _mean(num_microbatches: int) -> Accumulator:
  """An Accumulator that computes the mean of microbatched outputs."""
  if num_microbatches <= 0:
    raise ValueError(f'{num_microbatches=} must be positive.')
  return _lift(
      Accumulator(
          init=_with_floating_check(_identity),
          update=lambda carry, value, _: carry + value,
          finalize=lambda carry: carry / num_microbatches,
          aggregate=functools.partial(jnp.mean, axis=0),
      )
  )


def _running_mean() -> Accumulator:
  """An Accumulator that computes the running mean of microbatched outputs."""
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


def _get_out_sharding(x):
  """Compute the desired sharding of x.reshape(-1, *x.shape[2:], order='F')."""
  # We use dict because jax doesn't have out_sharding in older jax versions.
  if  jax.__version__ < '0.7.0':
    return {}
  sharding = jax.typeof(x).sharding
  if sharding.mesh.are_all_axes_explicit:
    if sharding.spec:
      # The first axis is not sharded, so we simply drop it.
      spec = jax.sharding.PartitionSpec(*sharding.spec[1:])
    else:
      spec = jax.sharding.PartitionSpec()
    return {'out_sharding': jax.sharding.NamedSharding(sharding.mesh, spec)}
  return {}


def _concat(num_microbatches: int) -> Accumulator:
  """An Accumulator that concatenates microbatched outputs along the axis 0."""
  if num_microbatches <= 0:
    raise ValueError(f'{num_microbatches=} must be positive.')

  def init(value):
    shape = (num_microbatches,) + value.shape
    zeros = jnp.broadcast_to(jnp.zeros_like(value), shape)
    return zeros.at[0].set(value)

  def update(carry, value, index):
    return carry.at[index].set(value)

  def finalize(carry):
    kwargs = _get_out_sharding(carry)
    return carry.reshape(-1, *carry.shape[2:], order='F', **kwargs)

  return _lift(Accumulator(init, update, finalize, _identity))


class AccumulationType(enum.Enum):
  """The type of accumulation to perform."""
  MEAN = enum.auto()
  """Average the microbatch outputs."""
  SUM = enum.auto()
  """Sum the microbatch outputs."""
  RUNNING_MEAN = enum.auto()
  """Average the microbatch outputs over `num_real_microbatches`."""
  CONCAT = enum.auto()
  """Concatenate the microbatch outputs along axis 0."""


# In order to construct some accumulators (MEAN, CONCAT), we need to know the
# number of microbatches. But we don't want to force the user to specify that in
# advance, so we offer an enum-based API for specifying accumulation strategies.
def _canonicalize(
    tree: Accumulator | AccumulationType | AccumulatorTree,
    num_microbatches: int | None,
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


def _reshape_all_args(
    microbatch_size: int,
    argnums: Sequence[int],
    argnames: Sequence[str],
    in_axes: Sequence[int],
    args: tuple[Any, ...],
    kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any], int]:
  """Reshapes all batch arguments to have a microbatch axis."""
  new_args = list(args)
  new_kwargs = dict(kwargs)
  batch_args = [args[i] for i in argnums] + [kwargs[i] for i in argnames]

  batch_sizes = jax.tree.flatten(jax.tree.map(
      lambda ax, subtree: jax.tree.map(lambda x: x.shape[ax], subtree),
      tuple(in_axes), tuple(batch_args)
  ))[0]

  if len(set(batch_sizes)) > 1:
    raise ValueError(
        f'Batch Arguments must have equal-size batch axes, found {batch_sizes}.'
    )

  batch_size = list(batch_sizes)[0]
  if batch_size % microbatch_size != 0:
    raise ValueError(f'{batch_size=} must be divisible by {microbatch_size=}.')

  for i, ax in zip(argnums, in_axes):
    new_args[i] = reshape_batch_axis(args[i], microbatch_size, ax)

  for name, ax in zip(argnames, in_axes[len(argnums) :]):
    new_kwargs[name] = reshape_batch_axis(kwargs[name], microbatch_size, ax)

  return tuple(new_args), new_kwargs, tuple(batch_sizes)[0]


def microbatch(
    fun: Function,
    argnums: int | Sequence[int],
    microbatch_size: int | None,
    accumulator: (
        Accumulator | AccumulationType | AccumulatorTree
    ) = AccumulationType.SUM,
    *,
    argnames: str | Sequence[str] = (),
    in_axes: int | Sequence[int] = 0,
    # TODO(mckennar): Document this option via notebook or doctest.
    num_real_microbatches: int | jax.Array | None = None,
) -> Function:
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
      fun: An arbitrary function.
      argnums: A sequence of argument indices that have a batch axis.
      microbatch_size: The number of rows in the overall batch used in each
        microbatch. Smaller values reduce memory overhead, but require more
        sequential computation. This must evenly divide the batch axis size of
        the batch arguments.
      accumulator: Specifies how to combine results from each microbatch; can be
        a single `Accumulator`, a pytree matching the structure of `fun`'s
        output, with `Accumulator` values at the leaves, or anything in between
        (i.e., a PyTree prefix of `fun`'s output`).
      argnames: A sequence of keyword argument names that have a batch axis.
      in_axes: An integer or sequence of integers indicating the batch axis
        index for each argument in `argnums` and `argnames` should be aligned
        with the list `argnums + argnames`. The default value of 0 assumes
        that all arguments have a batch axis on the 0th dimension of the array.
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

  if isinstance(argnames, str):
    argnames = (argnames,)

  if isinstance(in_axes, int):
    in_axes = (in_axes,) * (len(argnums) + len(argnames))

  def microbatched_fun(*args, **kwargs):
    reshaped_args, reshaped_kwargs, batch_size = _reshape_all_args(
        microbatch_size, argnums, argnames, in_axes, args, kwargs
    )
    num_microbatches = batch_size // microbatch_size
    accumulator_ = _canonicalize(accumulator, num_microbatches)

    def f(index):
      input_args = list(reshaped_args)
      input_kwargs = dict(reshaped_kwargs)
      for i, ax in zip(argnums, in_axes):
        input_args[i] = jax.tree.map(
            functools.partial(jnp.take, indices=index, axis=ax), input_args[i]
        )
      for i, ax in zip(argnames, in_axes[len(argnums) :]):
        input_kwargs[i] = jax.tree.map(
            functools.partial(jnp.take, indices=index, axis=ax), input_kwargs[i]
        )
      return fun(*input_args, **input_kwargs)

    def body_fun(index, carry):
      return accumulator_.update(carry, f(index), index)

    loop_bound = num_real_microbatches or num_microbatches
    answer = jax.lax.fori_loop(
        1, loop_bound, body_fun, accumulator_.init(f(0)),
    )

    return accumulator_.finalize(answer)

  return microbatched_fun


# TODO(mckennar): Create a notebook demonstrating useful use-cases.
def micro_vmap(
    fun: Function,
    in_axes: int | Sequence[int] = 0,
    out_axes: Any = 0,
    *,
    microbatch_size: int | None = None,
    vmap_fn: VmapFn = jax.vmap,
    accumulator: (
        Accumulator | AccumulationType | AccumulatorTree
    ) = AccumulationType.CONCAT,
) -> Function:
  """A generalized version of jax.vmap that supports microbatching.

  Because this function incorporates microbatching, you can vmap over
  arrays with much larger batch axis sizes than jax.vmap without running
  out of memory. This function generalizes vmap by introducing new keyword
  arguments `microbatch_size` and `accumulator` to control microbatching
  behavior. It specializes vmap by imposing stricter requirements on `in_axes`
  and `out_axes`.

  Example Usage:
    >>> from optax.experimental import microbatching
    >>> import jax.numpy as jnp
    >>> microbatching.micro_vmap(lambda x: x**2)(jnp.arange(8))
    Array([ 0,  1,  4,  9, 16, 25, 36, 49], dtype=int32)

  Args:
    fun: Function to be mapped over additional axes.
    in_axes: Array axis to map over.  See jax.vmap for more details.
    out_axes: Unsupported by optax.vmap, must be set to 0.
    microbatch_size: The number of rows in the overall batch used in each
      microbatch. Smaller values reduces memory overhead, but require more
      sequential computation. This must evenly divide the batch axis size of
      the batch arguments.
    vmap_fn: A function with the same signature as jax.vmap.  Can be used to
      e.g., pass in kwargs to vmap.
    accumulator: Specifies what to do with the vmapped outputs.  The default
      value (CONCAT) returns each output with a batch axis, matching the
      behavior of jax.vmap. Reductions over the batch axis are also possible,
      including MEAN and SUM, and can be used when the the full output with a
      batch axis is not needed and is too large to fit in memory. This
      accumulator can be any PyTree prefix of the outputs of `fun` to apply
      different reductions to different sub-trees.

  Returns:
    A new function with the same args and kwargs having an additional
    batch axis (according to in_axes).
  """

  if out_axes != 0:
    raise NotImplementedError('out_axis != 0 is not currently supported')

  if isinstance(in_axes, int):
    in_axes = (in_axes,)

  def vmap_reduce_fn(*args, **kwargs):
    output = vmap_fn(fun, in_axes, out_axes)(*args, **kwargs)
    microbatch_size_ = jax.tree.leaves(output)[0].shape[0]

    # We are only relying on the `aggregate` attribute of the accumulator, which
    # does not require knowledge of the number of microbatches.
    temporary_accumulator = _canonicalize(accumulator, microbatch_size_)
    return temporary_accumulator.aggregate(output)

  return microbatch(
      vmap_reduce_fn,
      argnums=tuple(x[0] for x in enumerate(in_axes) if x[1] is not None),
      microbatch_size=microbatch_size,
      accumulator=accumulator,
      in_axes=tuple(ax for ax in in_axes if ax is not None),
  )


def _normalize_fun_to_return_aux(fun, has_aux):
  if has_aux:
    return fun
  else:
    return lambda *args, **kwargs: (fun(*args, **kwargs), None)


def _with_extra_batch_axis(
    fun: Function, batch_argnums: Sequence[int]
) -> Function:
  """Wraps a function to add an extra batch axis to the batch_argnums."""
  def wrapped_fun(*args, **kwargs):
    args_with_group_axis = list(args)
    for i in batch_argnums:
      args_with_group_axis[i] = jax.tree.map(
          lambda x: jnp.expand_dims(x, axis=1), args[i]
      )
    return fun(*args_with_group_axis, **kwargs)

  return wrapped_fun


def micro_grad(
    fun: Function,
    has_aux: bool = False,
    argnums: int | Sequence[int] = 0,
    *,
    batch_argnums: int | Sequence[int] = 1,
    keep_batch_dim: bool = True,
    microbatch_size: int | None = None,
    accumulator: (
        Accumulator | AccumulationType | AccumulatorTree
    ) = AccumulationType.SUM,
    transform_fn: Callable[[chex.ArrayTree], chex.ArrayTree] = lambda x: x,
    metrics_fn: Callable[[chex.ArrayTree], chex.ArrayTree] = lambda x: None
) -> ValueAndGradFn:
  """Create a function to compute, transform, and sum per-example gradients.

  This function is similar to jax.value_and_grad, but works at the level of
  size-1 batches.  This function is defined in terms of general transformations
  transform_fn and metrics_fn which  can be useful to e.g.,
  * limit the effect of outlier batch elements by clipping per-example grads.
  * compute moments of the gradients on a per-example basis.
  * computing scalar or low-dimensional gradient metrics on a per-example basis.

  Other notable differences between this function and jax.value_and_grad:
  * at least one argument to `fun` must have a batch axis, and that argument
    should be passed to `batch_argnums`. The default value of `1` assumes that
    `fun` has the signature `fun(params, batch, ...)`.
  * The return signature is different. The gradient is always returned as the
    first output, while all auxiliary outputs are returned as a namedtuple in
    the second output (including values, function aux, and metrics).
  * This function may be able to work for far larger batch sizes than
    native jax.value_and_grad due to the built-in microbatching.

  Example Usage (see https://arxiv.org/abs/2510.00236):
    >>> from optax.experimental import microbatching
    >>> def mean_squared_loss(params, features, targets):
    ...   preds = features @ params
    ...   diff = preds - targets
    ...   return 0.5 * jnp.mean(diff**2)
    >>> params = jnp.zeros(1)
    >>> features = jnp.ones((4, 1))
    >>> targets = jnp.array([0, 2, 4, 6])
    >>> (grads, squared_grads), aux = microbatching.micro_grad(
    ...     mean_squared_loss,
    ...     argnums=0,
    ...     batch_argnums=(1,2),
    ...     accumulator=microbatching.AccumulationType.MEAN,
    ...     transform_fn=lambda x: (x, x**2),
    ...     metrics_fn=jnp.linalg.norm
    ... )(params, features, targets)
    >>> grads, squared_grads  # per-example grads are [0, 2, 4, 6]
    (Array([-3.], dtype=float32), Array([14.], dtype=float32))
    >>> aux.values
    Array([ 0.,  2.,  8., 18.], dtype=float32)
    >>> aux.metrics
    Array([0., 2., 4., 6.], dtype=float32)


  Args:
    fun: The function to compute the gradient of.
    has_aux: Whether the function returns auxiliary output.
    argnums: The indices of argument(s) to differentiate with respect to.
    batch_argnums: The indices of argument(s) with a batch axis.
    keep_batch_dim: Whether `fun` expects inputs to have a batch dimension.
    microbatch_size: The size of the microbatches to use when computing the
      per-example gradients. See `microbatch` for more details.
    accumulator: Specifies how to combine or aggregate the transformed gradients
      across the batch axis.
    transform_fn: A function to apply to per-example gradients before averaging.
    metrics_fn: A function to apply to per-example gradients before
      transforming. Will be returned on a per-example basis as part of the
      auxiliary output, and therefore should be scalar or low-dimensional.

  Returns:
    A function that computes the value and gradient of `fun`, averaging the
    results over microbatches and applying the `transform_fn` and `metrics_fn`
    as described above. The auxiliary output (including values, metrics,
    function aux) will all be returned on a per-example-basis.
  """

  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)

  fun = _normalize_fun_to_return_aux(fun, has_aux)
  value_and_grad_fn = jax.value_and_grad(fun, argnums, has_aux=True)

  def grad_fn(*args, **kwargs):
    value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
    result = transform_fn(grad)
    aux = IndividualOutputs(
        values=value_and_aux[0],
        metrics=metrics_fn(grad),
        aux=value_and_aux[1],
    )
    return result, aux

  in_axes = [None]*(max(batch_argnums) + 1)
  for i in batch_argnums:
    in_axes[i] = 0
  micro_fun = micro_vmap(
      grad_fn,
      in_axes=in_axes,
      accumulator=(accumulator, AccumulationType.CONCAT),
      microbatch_size=microbatch_size
  )
  if keep_batch_dim:
    micro_fun = _with_extra_batch_axis(micro_fun, batch_argnums)
  return micro_fun
