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
"""Utilities to access second-order oracles efficiently."""

import functools
from typing import Any, Callable, NamedTuple, Union

import chex
import jax
from optax._src import base
from optax._src import utils

Scalar = Union[jax.Array, float]


def make_hvp_fn(
    fn: Callable[..., Union[Scalar, tuple[Scalar, Any]]],
    params: base.Params,
    has_aux: bool = False,
    **fn_kwargs,
) -> tuple[
    Union[tuple[Scalar, Any], Scalar],
    base.Updates,
    Callable[[base.Params], base.Params],
]:
  r"""Instantiates Hessian vector product (hvp) function.

  In its simplest usage (for ``has_aux=False`` and ``**fn_kwargs`` empty),
  this method  returns the value of the function ``fn`` at the current
  ``params``, the gradient of ``fn`` at ``params``, and a function ``hvp_fn``
  that gives acces to Hessian vector products.
  In equation, this method returns

  .. math::

    (f(w), \nabla f(w), v \rightarrow \nabla^2 f(w)v),

  where :math:`f` denotes the function ``fn``, and  :math:`w` the parameters
  ``params``. The output :math:`v \rightarrow \nabla^2 f(w)v` gives access to
  Hessian vector products on any tangent vector :math:`v`.

  For ``has_aux=False`` and ``**fn_kwargs`` not empty, this method returns
  :math:`(f(w; x), \nabla f(w; x), v \rightarrow \nabla^2 f(w; x)v)`
  for :math:`x` some additional inputs fed into the method by keyword-only
  arguments given in ``**fn_kwargs``.

  For ``has_aux=True`` and ``**fn_kwargs`` not empty, this method  returns
  :math:`((f(w; x), a), \nabla f(w; x), v \rightarrow \nabla^2 f(w; x)v)`
  where :math:`a` is some auxiliary outputs returned by the function.

  Examples:
    >>> import jax.numpy as jnp
    >>> from optax import second_order
    >>> # Example without auxiliary output
    >>> def fn(params, x, y):
    ...   logits = x.dot(params)
    ...   return 0.5*jnp.sum((logits - y)**2)
    >>> params = jnp.array([1., 2., 3.])
    >>> x = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    >>> y = jnp.array([1., 2.])
    >>> value, grad, hvp_fn = second_order.make_hvp_fn(fn, params, x=x, y=y)
    >>> print(value, grad)
    534.5 [133. 176. 219.]
    >>> tangents = jnp.array([1., 2., 3.])
    >>> hvp = hvp_fn(tangents)
    >>> print(hvp)
    [142. 188. 234.]
    >>> # Example with auxiliary outputs
    >>> def fn_with_aux(params, x, y):
    ...   logits = x.dot(params)
    ...   return 0.5*jnp.sum((logits - y)**2), logits
    >>> (value, aux), grad, hvp_fn = second_order.make_hvp_fn(
    ...    fn_with_aux, params, x=x, y=y, has_aux=True
    ... )
    >>> print(aux)
    [14. 32.]

  .. note::
    This function is akin to :func:`jax.vjp` in the sense that it instantiates
    the hvp function rather than directly computing the hvp value. As a vjp, it
    stores in memory some intermediate computations to give access to the hvp
    function.
    When the hvp needs to be accessed multiple times, this function can reuse
    the information stored memory so that the function is not reevaluated each
    time.

  .. seealso:: :func:`optax.second_order.hvp_call`


  Args:
    fn: function to compute Hessian vector products from. (must be twice
      differentiable in JAX) Must return either a scalar (if ``has_aux =
      False``) or a pair of (scalar, aux) with aux some auxiliary output (if
      ``has_aux = True``).
    params: pytree of parameters at which we define the hvp.
    has_aux: whether the function has auxiliary outputs or not.
    **fn_kwargs: additional parameters for the function in keyword format.

  Returns:
    ``(value, grad, hvp_fn)`` if ``has_aux = False``.

    ``((value, aux), grad, hvp_fn)`` if ``has_aux=True``.

    where

    - ``value`` is the value of ``fn`` at ``params`` and ``**fn_kwargs``.

    - ``grad`` is the gradient of ``fn`` at ``params`` and ``**fn_kwargs``.

    - ``hvp_fn`` is a function that takes some pytree of tangent direction
    ``tangent`` of  the same shape as ``params`` and returns the product
    ``hvp_fn(tangent)`` of the Hessian of ``fn`` at ``params`` and
    ``**fn_kwargs`` with the ``tangent`` direction.

    - ``aux`` is some auxiliary output returned by the function.
  """

  def grad_and_value_fn(x):
    value_and_maybe_aux, grad = jax.value_and_grad(fn, has_aux=has_aux)(
        x, **fn_kwargs
    )
    return grad, value_and_maybe_aux

  grad, hvp_fn, value_and_maybe_aux = jax.linearize(
      grad_and_value_fn, params, has_aux=True
  )
  return value_and_maybe_aux, grad, hvp_fn


def hvp_call(
    fn: Callable[..., Union[Scalar, tuple[Scalar, Any]]],
    params: base.Params,
    tangents: base.Params,
    has_aux: bool = False,
    **fn_kwargs,
) -> tuple[Union[Scalar, tuple[Scalar, Any]], base.Updates, base.Params]:
  r"""Computes hessian vector product (hvp) directly by jvp on top of gradient.

  In its simplest usage (for ``has_aux=False`` and ``**fn_kwargs`` empty),
  this method  returns the value of the function ``fn`` at the current
  ``params``, the gradient of ``fn`` at ``params``, and the Hessian of ``fn``
  at params multipled by the ``tangents`` direction.
  In equation, this method returns

  .. math::

    (f(w), \nabla f(w), \nabla^2 f(w)v)

  where :math:`f` denotes the function ``fn``, :math:`w` the parameters
  ``params``, and :math:`v` the tangent direction ``tangent``.

  For ``has_aux=False`` and ``**fn_kwargs`` not empty, this method returns
  :math:`(f(w; x), \nabla f(w; x), \nabla^2 f(w; x)v)`,
  for :math:`x` some additional inputs fed into the method by keyword-only
  arguments given in ``**fn_kwargs``.

  For ``has_aux=True`` and ``**fn_kwargs`` not empty, this method returns
  :math:`((f(w; x), a), \nabla f(w; x), \nabla^2 f(w; x)v)`,
  where :math:`a` is some auxiliary outputs returned by the function.

  Examples:
    >>> import jax.numpy as jnp
    >>> from optax import second_order
    >>> # Example without auxiliary output
    >>> def fn(params, x, y):
    ...   logits = x.dot(params)
    ...   return 0.5*jnp.sum((logits - y)**2)
    >>> params = jnp.array([1., 2., 3.])
    >>> tangents = jnp.array([1., 2., 3.])
    >>> x = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    >>> y = jnp.array([1., 2.])
    >>> value, grad, hvp = second_order.hvp_call(
    ...   fn, params, tangents, x=x, y=y
    ... )
    >>> print(value, grad)
    534.5 [133. 176. 219.]
    >>> print(hvp)
    [142. 188. 234.]
    >>> # Example with auxiliary outputs
    >>> def fn_with_aux(params, x, y):
    ...   logits = x.dot(params)
    ...   return 0.5*jnp.sum((logits - y)**2), logits
    >>> (value, aux), grad, hvp = second_order.hvp_call(
    ...    fn_with_aux, params, tangents, x=x, y=y, has_aux=True
    ... )
    >>> print(aux)
    [14. 32.]

  .. note::
    This function is akin to :func:`jax.jvp` in the sense that it directly
    returns the desired hvp value. There is no lingering memory cost as with
    a :func:`jax.vjp` or in :func:`optax.second_order.make_hvp_fn`.
    However, when the hvp needs to be accessed multiple times, this method
    will reevaluate the function as many times. In other words,
    :func:`optax.second_order.make_hvp_fn` may be preferred when the Hessian
    must be accessed multiple times while :func:`optax.second_order.hvp_call`
    may be preferred for a single call the hvp

  .. seealso:: :func:`optax.second_order.make_hvp_fn`

  Args:
    fn: function to compute Hessian vector products from (must be twice
      differentiable in JAX) Must return either a scalar (if ``has_aux =
      False``) or a pair of (scalar, aux) with aux some auxiliary output (if
      ``has_aux = True``).
    params: pytree of parameters at which we define the hvp.
    tangents: pytree of tangent direction along which we compute the hvp. of the
      same shape as ``params``.
    has_aux: whether the function has auxiliary outputs or not.
    **fn_kwargs: additional parameters for the function in keyword format.

  Returns:
    ``(value, grad, hvp)`` if ``has_aux = False``.

    ``((value, aux), grad, hvp)`` if ``has_aux=True``.

    where

    - ``value`` is the value of ``fn`` at ``params`` and ``**fn_kwargs``.

    - ``grad`` is the gradient of ``fn```` at ``params`` and ``**fn_kwargs``.

    - ``hvp`` is the product of the Hessian of ``fn`` at ``params`` and
    ``**fn_kwargs`` with the ``tangent`` direction.

    - ``aux`` is some auxiliary output returned by the function.
  """

  def grad_and_value_fn(x):
    value_and_maybe_aux, grad = jax.value_and_grad(fn, has_aux=has_aux)(
        x, **fn_kwargs
    )
    return grad, value_and_maybe_aux

  grad, hvp, value_and_maybe_aux = jax.jvp(  # pylint: disable=unbalanced-tuple-unpacking
      grad_and_value_fn, (params,), (tangents,), has_aux=True
  )
  return value_and_maybe_aux, grad, hvp


class InnerOuterAux(NamedTuple):
  """Auxiliary outputs of two functions inner_fn, outer_fn."""

  inner: Any
  outer: Any


def make_gnvp_fn(
    inner_fn: Callable[..., Union[chex.ArrayTree, tuple[chex.ArrayTree, Any]]],
    outer_fn: Callable[..., Union[Scalar, tuple[Scalar, Any]]],
    params: base.Params,
    inner_fn_has_aux: bool = False,
    outer_fn_has_aux: bool = False,
    **fn_kwargs,
) -> tuple[
    Union[Scalar, tuple[Scalar, InnerOuterAux]],
    base.Updates,
    Callable[[base.Params], base.Params],
]:
  r"""Instantiates Gauss-Newton vector product (gnvp).

  For a composition :math:`f\circ g`, where :math:`g` is ``inner_fn`` and
  :math:`f` is ``outer_fn``, this method computes for
  ``inner_fn_has_aux=False``, ``outer_fn_has_aux=False``, and empty
  ``**fn_kwargs``,

  .. math::

    (f(g(w)), \nabla (f\circ g)(w), 
    v \rightarrow J_g(w)^\top H_f(z) J_g(w) v )),

  where :math:`J_g(w)` is the Jacobian of :math:`g` at :math:`w`,
  :math:`H_f(z)` is the Hessian of :math:`f` at :math:`z = g(w)`.
  The output :math:`v \rightarrow J_g(w)^T H_f(z) J_g(w)` gives access to
  Gauss-Newton vector products on any tangent vector :math:`v`.

  If ``**fn_kwargs`` is not empty, the method splits ``**fn_kwargs`` into
  keyword-only arguments ``inner_fn_kwargs`` and ``outer_fn_kwargs``
  by examining the signatures of ``inner_fn`` and ``outer_fn``.
  The method then returns (still for
  ``inner_fn_has_aux=False``, ``outer_fn_has_aux=False``)

  .. math::

    (f(g(w; x); y), \nabla (f(\cdot; y) \circ g(\cdot; x))(w),
    v \rightarrow J_g(w; x)^\top H_f(z; y) J_g(w; x) v )),

  where :math:`x` and :math:`y` are ``inner_fn_kwargs`` and ``outer_fn_kwargs``
  respectively.

  If ``inner_fn_has_aux=True`` or ``outer_fn_has_aux=True``, this method returns
  (presented for ``**fn_kwargs`` empty for simplicity)

  .. math::
    ((f(g(w)), (a_i, a_o)), \nabla (f\circ g)(w),
    v \rightarrow J_g(w)^\top H_f(z) J_g(w) v )),

  where :math:`a_i` and :math:`a_o` are the auxiliary outputs returned by,
  respectively ``inner_fn`` and ``outer_fn``. If e.g. ``inner_fn_has_aux=True``
  and ``outer_fn_has_aux=False``, the function still returns :math:`(a_i, a_o)`
  but with :math:`a_o` ``None``.

  Examples:
    >>> import jax.numpy as jnp
    >>> from optax import second_order
    >>> # Example without auxiliary output
    >>> def net(params, x):
    ...   return x.dot(params)
    >>> def loss(logits, y):
    ...   return 0.5*jnp.sum((logits - y)**2)
    >>> params = jnp.array([1., 2., 3.])
    >>> x = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    >>> y = jnp.array([1., 2.])
    >>> value, grad, gnvp_fn = second_order.make_gnvp_fn(
    ...   net, loss, params, x=x, y=y
    ... )
    >>> print(value, grad)
    534.5 [133. 176. 219.]
    >>> tangents = jnp.array([1., 2., 3.])
    >>> gnvp = gnvp_fn(tangents)
    >>> print(gnvp)
    [142. 188. 234.]
    >>> # Note that the same result would be obtained using hvp_fn since net
    >>> # (the inner_fn) is linear here.
    >>> # Example with auxiliary outputs
    >>> def loss_with_aux(logits, y):
    ...   return 0.5*jnp.sum((logits - y)**2), logits
    >>> (value, aux), grad, gnvp_fn = second_order.make_gnvp_fn(
    ...    net, loss_with_aux, params, x=x, y=y, outer_fn_has_aux=True
    ... )
    >>> print(aux)
    InnerOuterAux(inner=None, outer=Array([14., 32.], dtype=float32))

  Args:
    inner_fn: inner function of the composition, :math:`g` in the formula above.
      Must be differentiable in JAX. Can return a pytree (if ``inner_fn_has_aux
      = False``) or a pair of (pytree, aux) with aux some auxiliary output (if
      ``inner_fn_has_aux = True``). The output of inner_fn must match the first
      argument of outer_fn.
    outer_fn: outer function of the composition, :math:`f` in the formula above.
      Must be twice differentiable in JAX. Must return either a scalar (if
      ``outer_fn_has_aux = False``) or a pair of (scalar, aux) with aux some
      auxiliary output (if ``outer_fn_has_aux = True``).
    params: parameters of the composition, :math:`w` in the formula above.
    inner_fn_has_aux: whether the inner function returns auxiliary outputs.
    outer_fn_has_aux: whether the outer function returns auxiliary outputs.
    **fn_kwargs: additional parameters for the composition in keyword format. If
      ``**fn_kwargs`` is not empty, the method splits ``**fn_kwargs`` into
      keyword-only arguments ``inner_fn_kwargs`` and ``outer_fn_kwargs`` by
      examining the signatures of ``inner_fn`` and ``outer_fn``.

  Returns:
    ``(value, grad, gnvp_fn)`` if ``(inner_has_aux or outer_has_aux) = False``.

    ``((value, aux), grad, gnvp_fn)``
    if ``(inner_has_aux or outer_has_aux) = True``.

    where

    - ``value`` is the value of the composition of ``inner_fn`` and ``outer_fn``
    at ``params`` and ``**fn_kwargs``.

    - ``grad`` is the gradient of the composition of ``inner_fn`` and
    ``outer_fn`` at ``params`` and ``**fn_kwargs``.

    - ``gnvp_fn`` is a function that takes some pytree of tangent direction
    ``tangent`` of  the same shape as ``params`` and returns the product
    ``gnvp_fn(tangent)`` of the Gauss-Newton matrix defined from the composition
    of ``inner_fn`` and ``outer_fn`` evaluated at ``params`` and
    ``**fn_kwargs`` with a ``tangent`` direction.

    - ``aux`` is a ``NameTuple`` with entries ``inner``, ``outer`` for
    the auxiliary outputs of ``inner_fn`` and ``outer_fn`` respectively.
    If e.g. ``inner_fn_has_aux=True`` and ``outer_fn_has_aux=False``,
    then ``aux.outer`` is ``None``, or, equivalently, ``aux[1] = None``.
  """

  (inner_fn_kwargs, outer_fn_kwargs), remaining_kwargs = (
      utils._extract_fns_kwargs(  # pylint: disable=protected-access
          (inner_fn, outer_fn), fn_kwargs
      )
  )
  if remaining_kwargs:
    raise ValueError(
        f'Some arguments {remaining_kwargs} are not passed to inner_fn nor '
        'outer_fn.'
    )

  inner_fn_aux = None
  outer_fn_aux = None

  # Reduce inner fn to a single input function with auxiliary outputs
  inner_fn_ = functools.partial(inner_fn, **inner_fn_kwargs)

  # Instantiates jacobian vector product (jvp)
  if inner_fn_has_aux:
    outputs, inner_jvp_fn, inner_fn_aux = jax.linearize(
        inner_fn_, params, has_aux=True
    )
  else:
    outputs, inner_jvp_fn = jax.linearize(inner_fn_, params, has_aux=False)

  # Get jacobian transpose vector product (vjp) by linear transposition
  inner_vjp_fn_ = jax.linear_transpose(inner_jvp_fn, params)

  # jax.linear_transpose returns tuples (like a vjp), we won't deal with
  # multiple parameters so we alias it to return just what we want
  # (not a tuple (grad,), but just grad)
  inner_vjp_fn = lambda x: inner_vjp_fn_(x)[0]

  # Get hvp of outer function, with associated value, aux and gradient
  value_and_maybe_aux, outer_grad, outer_hvp_fn = make_hvp_fn(
      outer_fn, outputs, has_aux=outer_fn_has_aux, **outer_fn_kwargs
  )
  if outer_fn_has_aux:
    value, outer_fn_aux = value_and_maybe_aux
  else:
    value = value_and_maybe_aux

  # Compute overall gradient by backpropagating outer gradient through inner
  # vjp
  grad = inner_vjp_fn(outer_grad)

  # Creates gnvp fnction by adequate composition.
  # We make gnvp a valid Pytree to enable jitting make_gnvp (see tests).
  gnvp_fn = lambda tangents: inner_vjp_fn(outer_hvp_fn(inner_jvp_fn(tangents)))

  if inner_fn_has_aux or outer_fn_has_aux:
    return (value, InnerOuterAux(inner_fn_aux, outer_fn_aux)), grad, gnvp_fn
  else:
    return value, grad, gnvp_fn
