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
"""Line-searches."""

from collections.abc import Callable
import functools
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src import utils
import optax.tree


class BacktrackingLinesearchInfo(NamedTuple):
  """Information about the backtracking linesearch step, for debugging.

  Attributes:
    num_linesearch_steps: number of linesearch steps.
    decrease_error: error of the decrease criterion at the end of the
      linesearch. A positive value indicates that
      the linesearch failed to find a stepsize that ensures a sufficient
      decrease. A null value indicates it succeeded in finding such a stepsize.
  """
  num_linesearch_steps: Union[int, chex.Numeric]
  decrease_error: Union[float, chex.Numeric]


class ScaleByBacktrackingLinesearchState(NamedTuple):
  """State for :func:`optax.scale_by_backtracking_linesearch`.

  Attributes:
    learning_rate: learning rate computed at the end of a round of line-search,
      used to scale the update.
    value: value of the objective computed at the end of a round of line-search.
      Can be reused using :func:`optax.value_and_grad_from_state`.
    grad: gradient of the objective computed at the end of a round of
      line-search if the line-search is instantiated with store_grad = True.
      Otherwise it is None. Can be reused using
      :func:`optax.value_and_grad_from_state`.
    info: information about the backtracking linesearch step, for debugging.
  """

  learning_rate: Union[float, jax.Array]
  value: Union[float, jax.Array]
  grad: Optional[base.Updates]
  info: BacktrackingLinesearchInfo


class BacktrackingLineSearchState(NamedTuple):
  """State during the inner loop of a backtracking line-search."""

  learning_rate: Union[float, jax.Array]
  new_value: Union[float, jax.Array]
  new_grad: base.Updates
  decrease_error: chex.Numeric
  iter_num: Union[int, jax.Array]


def scale_by_backtracking_linesearch(
    max_backtracking_steps: int,
    slope_rtol: float = 1e-4,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
    store_grad: bool = False,
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
  r"""Backtracking line-search ensuring sufficient decrease (Armijo criterion).

  Selects learning rate :math:`\eta` such that it verifies the sufficient
  decrease criterion

  .. math::
    f(w + \eta u) \leq (1+\delta)f(w) + \eta c \langle u, \nabla f(w) \rangle +
    \epsilon \,,

  where

    :math:`f` is the function to minimize,
    :math:`w` are the current parameters,
    :math:`\eta` is the learning rate to find,
    :math:`u` is the update direction,
    :math:`c` is a coefficient (``slope_rtol``) measuring the relative decrease
    of the function in terms of the slope (scalar product between the gradient
    and the updates),
    :math:`\delta` is a relative tolerance (``rtol``),
    :math:`\epsilon` is an absolute tolerance (``atol``).

  The algorithm starts with a given guess of a learning rate and decrease it
  by ``decrease_factor`` until the criterion above is met.

  Args:
    max_backtracking_steps: maximum number of iterations for the line-search.
    slope_rtol: relative tolerance w.r.t. to the slope. The sufficient decrease
      must be slope_rtol * lr * <grad, updates>, see formula above.
    decrease_factor: decreasing factor to reduce learning rate.
    increase_factor: increasing factor to increase learning rate guess. Setting
      it to 1. amounts to keep the current guess, setting it to ``math.inf``
      amounts to start with ``max_learning_rate`` at each round.
    max_learning_rate: maximum learning rate (learning rate guess clipped to
      this).
    atol: absolute tolerance at which the criterion needs to be satisfied.
    rtol: relative tolerance at which the criterion needs to be satisfied.
    store_grad: whether to compute and store the gradient at the end of the
      linesearch. Since the function is called to compute the value to accept
      the learning rate, we can also access the gradient along the way. By doing
      that, we can directly reuse the value and the gradient computed at the end
      of the linesearch for the next iteration using
      :func:`optax.value_and_grad_from_state`. See the example above.
    verbose: whether to print debugging information.

  Returns:
    A :class:`GradientTransformationExtraArgs`, where the ``update`` function
    takes the following additional keyword arguments:

    * ``value``: value of the function at the current params.
    * ``grad``: gradient of the function at the current params.
    * ``value_fn``: function returning the value of the function we seek to
      optimize.
    * ``**extra_args``: additional keyword arguments, if the function needs
      additional arguments such as input data, they should be put there (
      see example in this docstring).

  Examples:

    An example on using the backtracking line-search with SGD::

      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> solver = optax.chain(
      ...    optax.sgd(learning_rate=1.),
      ...    optax.scale_by_backtracking_linesearch(max_backtracking_steps=15)
      ... )
      >>> # Function with additional inputs other than params
      >>> def fn(params, x, y): return optax.l2_loss(x.dot(params), y)
      >>> params = jnp.array([1., 2., 3.])
      >>> opt_state = solver.init(params)
      >>> x, y = jnp.array([3., 2., 1.]), jnp.array(0.)
      >>> xs, ys = jnp.tile(x, (5, 1)), jnp.tile(y, (5,))
      >>> opt_state = solver.init(params)
      >>> print('Objective function: {:.2E}'.format(fn(params, x, y)))
      Objective function: 5.00E+01
      >>> for x, y in zip(xs, ys):
      ...   value, grad = jax.value_and_grad(fn)(params, x, y)
      ...   updates, opt_state = solver.update(
      ...       grad,
      ...       opt_state,
      ...       params,
      ...       value=value,
      ...       grad=grad,
      ...       value_fn=fn,
      ...       x=x,
      ...       y=y
      ...   )
      ...   params = optax.apply_updates(params, updates)
      ...   print('Objective function: {:.2E}'.format(fn(params, x, y)))
      Objective function: 3.86E+01
      Objective function: 2.50E+01
      Objective function: 1.34E+01
      Objective function: 5.87E+00
      Objective function: 5.81E+00

    A similar example, but with a non-stochastic function where we can reuse
    the value and the gradient computed at the end of the linesearch:

      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> # Function without extra arguments
      >>> def fn(params): return jnp.sum(params ** 2)
      >>> params = jnp.array([1., 2., 3.])
      >>> # In this case we can store value and grad with the store_grad field
      >>> # and reuse them using optax.value_and_grad_state_from_state
      >>> solver = optax.chain(
      ...    optax.sgd(learning_rate=1.),
      ...    optax.scale_by_backtracking_linesearch(
      ...        max_backtracking_steps=15, store_grad=True
      ...    )
      ... )
      >>> opt_state = solver.init(params)
      >>> print('Objective function: {:.2E}'.format(fn(params)))
      Objective function: 1.40E+01
      >>> value_and_grad = optax.value_and_grad_from_state(fn)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...       grad, opt_state, params, value=value, grad=grad, value_fn=fn
      ...   )
      ...   params = optax.apply_updates(params, updates)
      ...   print('Objective function: {:.2E}'.format(fn(params)))
      Objective function: 5.04E+00
      Objective function: 1.81E+00
      Objective function: 6.53E-01
      Objective function: 2.35E-01
      Objective function: 8.47E-02

  References:
    Vaswani et al., `Painless Stochastic Gradient
    <https://arxiv.org/abs/1905.09997>`_, 2019

    Nocedal & Wright, `Numerical Optimization
    <https://doi.org/10.1007/978-0-387-40065-5>`_, 1999


  .. warning::
    The sufficient decrease criterion might be impossible to satisfy for some
    update directions. To guarantee a non-trivial solution for the sufficient
    decrease criterion, a descent direction for updates (:math:`u`) is required.
    An update (:math:`u`) is considered a descent direction if the derivative of
    :math:`f(w + \eta u)` at :math:`\eta = 0`
    (i.e.,  :math:`\langle u, \nabla f(w)\rangle`) is negative.  This condition
    is automatically satisfied when using :func:`optax.sgd` (without momentum),
    but may not hold true for other optimizers like :func:`optax.adam`.


    More generally, when chained with other transforms as
    ``optax.chain(opt_1, ..., opt_k,
    scale_by_backtraking_linesearch(max_backtracking_steps=...),
    opt_kplusone, ..., opt_n)``, the updates returned by chaining
    ``opt_1, ..., opt_k`` must be a descent direction. However, any transform
    after the backtracking line-search doesn't necessarily need to satisfy the
    descent direction property (one could for example use momentum).

  .. note:: The algorithm can support complex inputs.

  .. seealso:: :func:`optax.value_and_grad_from_state` to make this method
    more efficient for non-stochastic objectives.

  .. versionadded:: 0.2.0
  """

  def init_fn(params: base.Params) -> ScaleByBacktrackingLinesearchState:
    if store_grad:
      grad = optax.tree.zeros_like(params)
    else:
      grad = None
    # base output type on params type, except only real part if complex
    val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
    return ScaleByBacktrackingLinesearchState(
        learning_rate=jnp.array(1.0, dtype=val_dtype),
        value=jnp.array(jnp.inf, dtype=val_dtype),
        grad=grad,
        info=BacktrackingLinesearchInfo(
            num_linesearch_steps=jnp.asarray(0),
            decrease_error=jnp.array(jnp.inf, dtype=val_dtype),
        ),
    )

  def _compute_decrease_error(
      stepsize: chex.Numeric,
      slope: chex.Numeric,
      value: chex.Numeric,
      new_value: chex.Numeric,
  ) -> chex.Numeric:
    decrease_error = (
        new_value - (1.0 + rtol) * value - stepsize * slope_rtol * slope
    )
    decrease_error = jnp.where(
        jnp.isnan(decrease_error), jnp.inf, decrease_error
    )
    return jnp.maximum(decrease_error, 0.0)

  def update_fn(
      updates: base.Updates,
      state: ScaleByBacktrackingLinesearchState,
      params: base.Params,
      *,
      value: Union[float, jax.Array],
      grad: base.Updates,
      value_fn: Callable[..., Union[jax.Array, float]],
      **extra_args: dict[str, Any],
  ) -> tuple[base.Updates, ScaleByBacktrackingLinesearchState]:
    """Compute scaled updates guaranteeing decrease of current objective.

    Args:
      updates: current updates.
      state: current state.
      params: current parameters.
      value: value of the function at the current params.
      grad: gradient of the function at the current params.
      value_fn: function returning the value of the function we seek to
        optimize.
      **extra_args: additional keyword arguments, if the function needs
        additional arguments such as input data, they should be put there, see
        the example in the docstring of the transform.

    Returns:
      updates: updates for the params (new_params = params + updates).
      state: updated state.

    .. warning:: The objective to minimize, ``value_fn``, can take more than
        one input, but must return a single scalar (float or jax.Array of
        dimension one). If the function requires more than one input, the
        additional inputs need to be fed to the update, see the example in the
        docstring of the transform. The function value_fn needs to be amenable
        to differentiation in JAX.
    """
    # Fetch arguments to be fed to value_fn from the extra_args
    (fn_kwargs,), remaining_kwargs = utils._extract_fns_kwargs(  # pylint: disable=protected-access
        (value_fn,), extra_args
    )
    del remaining_kwargs

    # Slope of lr -> value_fn(params + lr * updates) at lr = 0
    # Should be negative to ensure that there exists a lr (potentially
    # infinitesimal) that satisfies the criterion.
    slope = optax.tree.real(optax.tree.vdot(updates, optax.tree.conj(grad)))

    def cond_fn(
        search_state: BacktrackingLineSearchState,
    ):
      """Whether to stop the line-search inner loop."""
      decrease_error = search_state.decrease_error
      iter_num = search_state.iter_num
      return (~(decrease_error <= atol)) & (iter_num <= max_backtracking_steps)

    def body_fn(
        search_state: BacktrackingLineSearchState,
    ) -> BacktrackingLineSearchState:
      """Line-search inner loop step."""
      learning_rate = search_state.learning_rate
      new_grad = search_state.new_grad
      iter_num = search_state.iter_num
      # We start decreasing the learning rate after the first iteration
      # and up until the criterion is satisfied.
      learning_rate = jnp.where(
          iter_num > 0, decrease_factor * learning_rate, learning_rate
      )
      new_params = optax.tree.add_scale(params, learning_rate, updates)

      value_fn_ = functools.partial(value_fn, **fn_kwargs)
      if store_grad:
        # We evaluate value_fn and get its jvp operator so that we can
        # compute the gradient by transposing the jvp.
        new_value, jvp_value_fn = jax.linearize(value_fn_, new_params)

        decrease_error = _compute_decrease_error(
            learning_rate, slope, value, new_value
        )
        # If the line-search ends, we get the gradient for the new round of
        # line-search.
        new_grad = jax.lax.cond(
            (decrease_error <= atol) | (iter_num == max_backtracking_steps),
            lambda p: jax.linear_transpose(jvp_value_fn, p)(1.0)[0],
            lambda *_: new_grad,
            new_params,
        )
      else:
        # Here we just compute the value and leave the gradient as is
        new_value = value_fn_(new_params)
        decrease_error = _compute_decrease_error(
            learning_rate, slope, value, new_value
        )
      new_search_state = BacktrackingLineSearchState(
          learning_rate=learning_rate,
          new_value=new_value,
          new_grad=new_grad,
          decrease_error=decrease_error,
          iter_num=iter_num + 1,
      )
      return optax.tree.cast_like(new_search_state, other_tree=search_state)

    # We start with a guess candidate learning rate that may be larger than
    # the current one but no larger than the maximum one.
    learning_rate = jnp.minimum(
        increase_factor * state.learning_rate, max_learning_rate
    )
    search_state = BacktrackingLineSearchState(
        learning_rate=learning_rate,
        new_value=value,
        new_grad=optax.tree.zeros_like(params),
        decrease_error=jnp.array(jnp.inf),
        iter_num=0,
    )
    search_state = jax.lax.while_loop(cond_fn, body_fn, search_state)

    # If store_grad is False we simply return None (to not mix up with
    # optax.tree.zeros_like(params))
    new_grad = search_state.new_grad if store_grad else None
    new_value = search_state.new_value
    # If the decrease error is infinite, we avoid making any step (which would
    # result in nan or infinite values): we set the learning rate to 0.
    new_learning_rate = jnp.where(
        jnp.isinf(search_state.decrease_error), 0., search_state.learning_rate
    )

    if verbose:
      # We print information only if the linesearch failed.
      _cond_print(
          search_state.decrease_error > atol,
          "INFO: optax.scale_by_backtracking_linesearch:\n"
          "Backtracking linesearch failed to find a stepsize ensuring sufficent"
          " decrease.\n"
          "Value at current params: {value},\n"
          "Slope along update direction: {slope}\n"
          "Stepsize: {stepsize}\n"
          "Decrease Error: {decrease_error}",
          stepsize=search_state.learning_rate,
          decrease_error=search_state.decrease_error,
          value=value,
          slope=slope,
      )
      _cond_print(
          jnp.isinf(search_state.decrease_error),
          "Using a stepsize of 0 to avoid infinite or nan values.",
      )
    # At the end, we just scale the updates with the learning rate found.
    new_updates = optax.tree.scale(new_learning_rate, updates)
    info = BacktrackingLinesearchInfo(
        num_linesearch_steps=search_state.iter_num,
        decrease_error=search_state.decrease_error,
    )
    new_state = ScaleByBacktrackingLinesearchState(
        learning_rate=new_learning_rate,
        value=new_value,
        grad=new_grad,
        info=info
    )

    return new_updates, optax.tree.cast_like(new_state, other_tree=state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def _cond_print(condition, message, **kwargs):
  """Prints message if condition is true."""
  jax.lax.cond(
      condition,
      lambda _: jax.debug.print(message, **kwargs, ordered=True),
      lambda _: None,
      None,
  )


# pylint: disable=invalid-name
def _cubicmin(a, fa, fpa, b, fb, c, fc):
  """Cubic interpolation.

  Finds a critical point of a cubic polynomial
  p(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D, that goes through
  the points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
  May return NaN (if radical<0), in that case, the point will be ignored.
  Adapted from scipy.optimize._linesearch.py.

  Args:
    a: scalar
    fa: value of a function f at a
    fpa: slope of a function f at a
    b: scalar
    fb: value of a function f at b
    c: scalar
    fc: value of a function f at c

  Returns:
    xmin: point at which p'(xmin) = 0
  """
  C = fpa
  db = b - a
  dc = c - a
  denom = (db * dc) ** 2 * (db - dc)
  d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
  A, B = (
      jnp.dot(
          d1,
          jnp.array([fb - fa - C * db, fc - fa - C * dc]),
          precision=jax.lax.Precision.HIGHEST,
      )
      / denom
  )

  radical = B * B - 3.0 * A * C
  xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

  return xmin


def _quadmin(a, fa, fpa, b, fb):
  """Quadratic interpolation.

  Finds a critical point of a quadratic polynomial
  p(x) = B*(x-a)^2 + C*(x-a) + D, that goes through
  the points (a,fa), (b,fb) with derivative at a of fpa.
  Adapted from scipy.optimize._linesearch.py.

  Args:
    a: scalar
    fa: value of a function f at a
    fpa: slope of a function f at a
    b: scalar
    fb: value of a function f at b

  Returns:
    xmin: point at which p'(xmin) = 0
  """
  D = fa
  C = fpa
  db = b - a
  B = (fb - D - C * db) / (db**2)
  xmin = a - C / (2.0 * B)
  return xmin


# pylint: enable=invalid-name


class ZoomLinesearchState(NamedTuple):
  """State of the zoom linesearch."""

  count: chex.Numeric

  # Fixed attributes
  params: base.Params  # current parameters
  updates: base.Updates  # update direction
  stepsize_guess: chex.Numeric  # initial guess on the stepsize

  # Changing attributes
  stepsize: chex.Numeric  # current stepsize
  value: chex.Numeric  # value at current stepsize
  grad: base.Updates  # gradient at current stepsize
  slope: chex.Numeric  # slope of function along updates at current stepsize

  # Initial information stored to compute the errors
  value_init: chex.Numeric
  slope_init: chex.Numeric

  # Stopping criterions measures
  decrease_error: chex.Numeric
  curvature_error: chex.Numeric
  error: chex.Numeric

  # Booleans to switch between the two phases of the algorithm or stop the
  # linesearch
  interval_found: chex.Numeric
  done: chex.Numeric
  failed: chex.Numeric

  # Set up after interval search done, modified during zoom
  # Here low, high refer to stepsizes defining an interval where a valid
  # stepsize can be found. However the terms low, high do not refer to
  # low being smaller than high but to value(high) >= value(low).
  low: chex.Numeric
  value_low: chex.Numeric
  slope_low: chex.Numeric
  high: chex.Numeric
  value_high: chex.Numeric
  slope_high: chex.Numeric
  cubic_ref: chex.Numeric
  value_cubic_ref: chex.Numeric

  # Safeguard point: we may not be able to satisfy the curvature criterion
  # but we can still return a point that satisfies the decrease criterion
  safe_stepsize: chex.Numeric
  safe_value: chex.Numeric
  safe_grad: base.Updates


def zoom_linesearch(
    max_linesearch_steps: int,
    max_stepsize: Optional[float] = None,
    tol: float = 0.0,
    increase_factor: float = 2.0,
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    approx_dec_rtol: Optional[float] = 1e-6,
    interval_threshold: float = 1e-5,
    verbose: bool = False,
) -> tuple[
    Callable[..., ZoomLinesearchState],
    Callable[..., ZoomLinesearchState],
    Callable[..., Union[bool, chex.Numeric]],
]:
  r"""Zoom Linesearch ensuring sufficient decrease and small curvature.

  This linesearch algorithm finds a step size that satisfies both a
  sufficient decrease criterion and a small curvature criterion.
  See :func:`optax.scale_by_zoom_linesearch`
  for a detailed mathematical description of these criteria.

  The algorithm proceeds in two phases:

  1. **Interval Search:**
    Finds an upper bound :math:`\bar\eta` on the step size such that there
    exists a step size :math:`\eta \in [0, \bar\eta]` satisfying both criteria.

  2. **Zoom (Bisection):**
    Searches within the initial interval to find a suitable step size.
    This phase uses a bisection-like method, informed by the problem properties
    and criteria. It iteratively narrows the interval until a satisfactory step
    size is found.

  Args:
    max_linesearch_steps: maximum number of linesearch iterations.
    max_stepsize: maximal admissible learning rate. Can be set to ``None`` for
      no upper bound. An inappropriate value may prevent the linesearch to find
      a learning rate satisfying the small curvature criterion, since the latter
      may require sufficiently large stepsizes.
    tol: tolerance on the criterions.
    increase_factor: increasing factor to augment the learning rate when
      searching for an initial valid interval (1st phase above).
    slope_rtol: relative tolerance for the slope in the sufficient decrease
      criterion, see :func:`optax.scale_by_zoom_linesearch`.
    curv_rtol: relative tolerance for the curvature in the small curvature
      criterion, see :func:`optax.scale_by_zoom_linesearch`.
    approx_dec_rtol: relative tolerance for the initial value in the approximate
      sufficient decrease criterion. Can be set to ``None`` to use only the
      Armijo-Goldstein decrease criterion, see
      :func:`optax.scale_by_zoom_linesearch`.
    interval_threshold: if the size of the interval searched is below this
      threshold and a sufficient decrease for some stepsize has been found, then
      the linesearch selects the stepsize and ends.
    verbose: whether to print debugging information if the linesearch fails.

  Returns:
    tuple (``init_fn``, ``step_fn``, ``cond_step_fn``) where

    * ``init_fn(updates, params, *, value, grad, stepsize_guess) ->
      ZoomLinesearchState``
      initializes the state of the linesearch given the current
      parameters ``params``, the ``updates`` direction, the ``value`` of the
      function at the ``params``, the gradient ``grad`` of the function at
      ``params
    * ``step_fn(state, value_and_grad_fn, fn_kwargs) -> ZoomLinesearchState``
      updates the state of the linesearch given the current ``state``, a
      function ``value_and_grad_fn`` returning the value and the gradient of the
      function at any given inputs, ``fn_kwargs`` additional keyword-only
      arguments to be passed to the function (other than parameters, it could be
      additional data for example)
    * ``cond_step_fn(state) -> bool`` returns a boolean indicating whether the
      linesearch iterations should continue (``True``) or not (``False``).

  References:
    Algorithms 3.5 3.6 of Nocedal and Wright, `Numerical Optimization
    <https://doi.org/10.1007/978-0-387-40065-5>`_, 1999

    Hager and Zhang `Algorithm 851: CG_DESCENT, a Conjugate Gradient Method
    with Guaranteed Descent
    <https://doi.org/10.1145/1132973.1132979>`_, 2006
  """

  def _value_and_slope_on_line(
      value_and_grad_fn: Callable[..., tuple[chex.Numeric, base.Updates]],
      params: base.Params,
      stepsize: chex.Numeric,
      updates: base.Updates,
      fn_kwargs,
  ) -> tuple[base.Params, chex.Numeric, base.Updates, chex.Numeric]:
    r"""Compute value and slope on line.

    Mathematically, outputs

    .. math::
      (w_\eta, f(w_\eta; x), \nabla f(w_\eta; x), \partial f(w_\eta; x)(u)

    for :math:`w_\eta = w + \eta u`,
    where
      - :math:`w` are the current parameters ``params``,
      - :math:`u` is the update direction ``updates``,
      - :math:`\eta` is the stepsize, a.k.a. learning rate, ``stepsize``,
      - :math:`x` are additional arguments to the function ``fn_kwargs``,
      - :math:`\nabla f(w_\eta; x)` is the gradient of :math:`f(\cdot; x)`
        at the new step, :math:`w_eta`,
      - :math:`\partial f(w_\eta; x)(u)` is the directional derivative of
        :math:`f(\cdot; x)` at :math:`w_\eta` in the direction :math:`u`,
        that is, the slope (derivative) of
        :math:`\eta \rightarrow f(w + \eta u)` at :math:`\eta`.
        So :math:`\partial f(w_\eta; x)(u) = \nabla f(w_\eta)^\top u`.

    Args:
      value_and_grad_fn: function returning value and gradient at given inputs
      params: current parameters
      stepsize: tentative stepsize taken
      updates: direction along which the step is taken
      fn_kwargs: additional arguments to be passed to the function

    Returns:
      ``(step, value_step, grad_step, slope_step)``
      where

        * ``step`` are the parameters at the selected stepsize,
        * ``value_step`` is the value at the step,
        * ``grad_step`` is the gradient at the step,
        * ``slope_step`` is the derivative of the function in terms of the
          stepsize at the step.
    """
    step = optax.tree.add_scale(params, stepsize, updates)
    value_step, grad_step = value_and_grad_fn(step, **fn_kwargs)
    slope_step = optax.tree.real(optax.tree.vdot(optax.tree.conj(grad_step),
                                                 updates))
    return step, value_step, grad_step, slope_step

  def _compute_decrease_error(
      stepsize: chex.Numeric,
      value_step: chex.Numeric,
      slope_step: chex.Numeric,
      value_init: chex.Numeric,
      slope_init: chex.Numeric,
  ) -> chex.Numeric:
    """Compute decrease error."""
    # We consider either the usual sufficient decrease (Armijo criterion), see
    # equation (3.7a) of [Nocedal and Wright, 1999]
    decrease_error = (
        value_step - value_init - slope_rtol * stepsize * slope_init
    )
    if approx_dec_rtol is not None:
      # or an approximate decrease criterion, see equation (23) of
      # [Hager and Zhang, 2006].
      approx_decrease_error = slope_step - (2 * slope_rtol - 1.0) * slope_init

      # The classical Armijo criterion may fail to be satisfied if we are too
      # close to a minimum, causing the optimizer to fail as explained in
      # [Hager and Zhang, 2006].

      # We switch to the approximate decrease criterion only if we are close
      # enough to the minimizer. To measure this we check whether the
      # new value is smaller than the initial one up to a tolerance of
      # the order of magnitude of the initial value (see equations (26) and
      # (27) of [Hager and Zhang, 2006] that we simplify using one iterate in
      # equation (26)).
      delta_values = (
          value_step - value_init - approx_dec_rtol * jnp.abs(value_init)
      )
      approx_decrease_error = jnp.maximum(approx_decrease_error, delta_values)
      # We take then the *minimum* of both errors.
      decrease_error = jnp.minimum(approx_decrease_error, decrease_error)

    # We only care whether the criterion is violated (larger than 0.0) and
    # take care of potential nan values by converting to inf
    decrease_error = jnp.maximum(decrease_error, 0.0)
    decrease_error = jnp.where(
        jnp.isnan(decrease_error), jnp.inf, decrease_error
    )
    return decrease_error

  def _compute_curvature_error(
      slope_step: chex.Numeric, slope_init: chex.Numeric
  ) -> chex.Numeric:
    """Compute curvature error."""
    # See equation (3.7b) of [Nocedal and Wright, 1999].
    curvature_error = jnp.abs(slope_step) - curv_rtol * jnp.abs(slope_init)

    # We only care whether the criterion is violated (larger than 0.0) and
    # take care of potential nan values by converting to inf
    curvature_error = jnp.maximum(curvature_error, 0.0)
    curvature_error = jnp.where(
        jnp.isnan(curvature_error), jnp.inf, curvature_error
    )
    return curvature_error

  def _try_safe_step(
      state: ZoomLinesearchState,
  ) -> ZoomLinesearchState:
    """Try making a step with stepsize ensuring at least sufficient decrease."""
    outside_domain = jnp.isinf(state.decrease_error)
    final_stepsize, final_value, final_grad = optax.tree.where(
        (state.safe_stepsize > 0.0) | outside_domain,
        [state.safe_stepsize, state.safe_value, state.safe_grad],
        [state.stepsize, state.value, state.grad],
    )
    if verbose:
      jax.debug.print(
          "INFO: optax.scale_by_zoom_linesearch:\n"
          "Value at current params: {value_init}\n"
          "Slope along update direction: {slope_init}\n"
          "Stepsize reached: {stepsize}\n"
          "Decrease Error: {decrease_error}\n"
          "Curvature Error: {curvature_error}",
          value_init=state.value_init,
          slope_init=state.slope_init,
          stepsize=state.stepsize,
          decrease_error=state.decrease_error,
          curvature_error=state.curvature_error,
          ordered=True,
      )
      interval_length = jnp.abs(state.low - state.high)
      too_small_int = interval_length <= interval_threshold
      _cond_print(
          too_small_int,
          FLAG_INTERVAL_TOO_SMALL + " Interval length: {interval_length}.",
          interval_length=interval_length,
      )
      jax.lax.cond(
          state.safe_stepsize > 0.0,
          lambda _: jax.debug.print(
              FLAG_CURVATURE_COND_NOT_SATISFIED
              + " Stepsize ensuring sufficient decrease: {safe_stepsize}.",
              safe_stepsize=state.safe_stepsize,
          ),
          _failure_diagnostic,
          state,
      )
    final_state = state._replace(
        stepsize=final_stepsize, grad=final_grad, value=final_value
    )
    return optax.tree.cast_like(final_state, other_tree=state)

  def _search_interval(
      state: ZoomLinesearchState,
      value_and_grad_fn: Callable[..., tuple[chex.Numeric, base.Updates]],
      fn_kwargs: dict[str, Any],
  ) -> ZoomLinesearchState:
    """Search initial interval, Algorithm 3.5 of [Nocedal and Wright, 1999]."""

    iter_num = state.count

    params = state.params
    updates = state.updates
    stepsize_guess = state.stepsize_guess

    value_init = state.value_init
    slope_init = state.slope_init

    prev_stepsize = state.stepsize
    prev_value_step = state.value
    prev_slope_step = state.slope

    safe_stepsize = state.safe_stepsize
    safe_value = state.safe_value
    safe_grad = state.safe_grad

    # Choose new point, larger than previous one or set to initial guess
    # for first iteration.
    larger_stepsize = increase_factor * prev_stepsize
    new_stepsize = jnp.where(iter_num == 0, stepsize_guess, larger_stepsize)
    if max_stepsize is not None:
      max_stepsize_reached = new_stepsize >= max_stepsize
      new_stepsize = jnp.minimum(new_stepsize, max_stepsize)
    else:
      max_stepsize_reached = jnp.asarray(False)

    _, new_value_step, new_grad_step, new_slope_step = _value_and_slope_on_line(
        value_and_grad_fn, params, new_stepsize, updates, fn_kwargs
    )

    decrease_error = _compute_decrease_error(
        new_stepsize, new_value_step, new_slope_step, value_init, slope_init
    )
    curvature_error = _compute_curvature_error(new_slope_step, slope_init)
    new_error = jnp.maximum(decrease_error, curvature_error)

    # If the new point satisfies at least the decrease error we keep it
    # in case the curvature error cannot be satisfied.
    safe_decrease = decrease_error <= tol
    new_safe_stepsize, new_safe_value, new_safe_grad = optax.tree.where(
        safe_decrease,
        [new_stepsize, new_value_step, new_grad_step],
        [safe_stepsize, safe_value, safe_grad],
    )

    # If the new point is not good, set high and low values according to
    # conditions described in Algorithm 3.5 of [Nocedal and Wright, 1999]
    set_high_to_new = (decrease_error > 0.0) | (
        (new_value_step >= prev_value_step) & (iter_num > 0)
    )
    set_low_to_new = (new_slope_step >= 0.0) & (~set_high_to_new)

    # By default we set high to new and correct if we should have set
    # low to new. If none should have set, the search for the interval
    # continues anyway.
    low_, value_low_, slope_low_, high_, value_high_, slope_high_ = (
        prev_stepsize,
        prev_value_step,
        prev_slope_step,
        new_stepsize,
        new_value_step,
        new_slope_step,
    )

    default = [low_, value_low_, slope_low_, high_, value_high_, slope_high_]
    candidate = [
        new_stepsize,
        new_value_step,
        new_slope_step,
        prev_stepsize,
        prev_value_step,
        prev_slope_step,
    ]
    [low, value_low, slope_low, high, value_high, slope_high] = (
        optax.tree.where(set_low_to_new, candidate, default)
    )

    # If high or low have been set or the point is good, the interval has been
    # found. Otherwise, we'll keep on augmenting the stepsize.
    interval_found = set_high_to_new | set_low_to_new | (new_error <= tol)

    # If new_error <= tol, the line search is done. If the maximal stepsize
    # is reached, either an interval has been found and we will zoom into this
    # interval or no interval has been found meaning that the maximal stepsize
    # satisfies the Armijo criterion but a priori not the curvature criterion.
    # In that case there is no hope to satisfy the curvature criterion as it
    # would a priori be found for a larger stepsize, so we simply take the
    # maximal stepsize and flag that we could not satisfy the curvature
    # criterion.
    done = (new_error <= tol) | (max_stepsize_reached & ~interval_found)
    if verbose:
      _cond_print(
          (max_stepsize_reached & ~interval_found),
          "INFO: optax.scale_by_zoom_linesearch:\n"
          "Value at current params: {value_init}\n"
          "Slope along update direction: {slope_init}\n"
          "Stepsize reached: {stepsize}\n"
          "Decrease Error: {decrease_error}\n"
          "Curvature Error: {curvature_error}"
          + FLAG_INTERVAL_NOT_FOUND
          + "\n"
          + FLAG_CURVATURE_COND_NOT_SATISFIED,
          value_init=value_init,
          slope_init=slope_init,
          stepsize=new_stepsize,
          decrease_error=decrease_error,
          curvature_error=curvature_error,
      )
    failed = (iter_num + 1 >= max_linesearch_steps) & (~done)

    new_state = ZoomLinesearchState(
        count=numerics.safe_increment(iter_num),
        #
        params=params,
        updates=updates,
        stepsize_guess=stepsize_guess,
        #
        stepsize=new_stepsize,
        value=new_value_step,
        grad=new_grad_step,
        slope=new_slope_step,
        #
        value_init=value_init,
        slope_init=slope_init,
        #
        decrease_error=decrease_error,
        curvature_error=curvature_error,
        error=new_error,
        #
        interval_found=jnp.asarray(interval_found),
        done=jnp.asarray(done),
        failed=jnp.asarray(failed),
        #
        low=low,
        value_low=value_low,
        slope_low=slope_low,
        high=high,
        value_high=value_high,
        slope_high=slope_high,
        cubic_ref=low,
        value_cubic_ref=value_low,
        #
        safe_stepsize=new_safe_stepsize,
        safe_value=new_safe_value,
        safe_grad=new_safe_grad,
    )
    return optax.tree.cast_like(new_state, other_tree=state)

  def _zoom_into_interval(
      state: ZoomLinesearchState,
      value_and_grad_fn: Callable[..., tuple[chex.Numeric, base.Updates]],
      fn_kwargs: dict[str, Any],
  ) -> ZoomLinesearchState:
    """Zoom procedure, Algorithm 3.6 of [Nocedal and Wright, 1999]."""

    iter_num = state.count

    params = state.params
    updates = state.updates

    value_init = state.value_init
    slope_init = state.slope_init

    low = state.low
    value_low = state.value_low
    slope_low = state.slope_low
    high = state.high
    value_high = state.value_high
    slope_high = state.slope_high
    cubic_ref = state.cubic_ref
    value_cubic_ref = state.value_cubic_ref

    safe_stepsize = state.safe_stepsize
    safe_value = state.safe_value
    safe_grad = state.safe_grad

    # Check if interval not too small otherwise fail
    delta = jnp.abs(high - low)
    left = jnp.minimum(high, low)
    right = jnp.maximum(high, low)
    cubic_chk = 0.2 * delta
    quad_chk = 0.1 * delta

    # We use rather large values of interval threshold compared to machine
    # precision such that we avoid wasting iterations to satisfy curvature
    # criterion (a stepsize reducing values is taken if it exists when threshold
    # is met)
    too_small_int = delta <= interval_threshold

    # Find new point by interpolation
    middle_cubic = _cubicmin(
        low, value_low, slope_low, high, value_high, cubic_ref, value_cubic_ref
    )
    middle_cubic_valid = (middle_cubic > left + cubic_chk) & (
        middle_cubic < right - cubic_chk
    )
    use_cubic = middle_cubic_valid
    middle_quad = _quadmin(low, value_low, slope_low, high, value_high)
    middle_quad_valid = (middle_quad > left + quad_chk) & (
        middle_quad < right - quad_chk
    )
    use_quad = (~use_cubic) & middle_quad_valid
    middle_bisection = (low + high) / 2.0
    use_bisection = (~use_cubic) & (~use_quad)

    middle = jnp.where(use_cubic, middle_cubic, cubic_ref)
    middle = jnp.where(use_quad, middle_quad, middle)
    middle = jnp.where(use_bisection, middle_bisection, middle)

    # Check if new point is good
    _, value_middle, grad_middle, slope_middle = _value_and_slope_on_line(
        value_and_grad_fn, params, middle, updates, fn_kwargs
    )

    decrease_error = _compute_decrease_error(
        middle, value_middle, slope_middle, value_init, slope_init
    )
    curvature_error = _compute_curvature_error(slope_middle, slope_init)
    new_error = jnp.maximum(decrease_error, curvature_error)

    # If the new point satisfies at least the decrease error we keep it in case
    # the curvature error cannot be satisfied.
    # We take the one with the smallest value.
    safe_decrease = decrease_error <= tol
    update_safe_stepsize = safe_decrease & (value_middle < safe_value)
    new_safe_stepsize, new_safe_value, new_safe_grad = optax.tree.where(
        update_safe_stepsize,
        [middle, value_middle, grad_middle],
        [safe_stepsize, safe_value, safe_grad],
    )

    # If both Armijo and curvature criterions are satisfied, we are done.
    # In any case, we take the stepizes, value and grad computed at the new
    # middle point for the running state.
    done = new_error <= tol

    # Otherwise, we update high and low values
    set_high_to_middle = (decrease_error > 0.0) | (value_middle >= value_low)
    secant_interval = slope_middle * (high - low)
    set_high_to_low = (secant_interval >= 0.0) & (~set_high_to_middle)
    set_low_to_middle = ~set_high_to_middle

    # Set high to middle, or low, or keep as it is
    default = [high, value_high, slope_high]
    candidate = [middle, value_middle, slope_middle]
    [new_high_, new_value_high_, new_slope_high_] = optax.tree.where(
        set_high_to_middle, candidate, default
    )
    default = [new_high_, new_value_high_, new_slope_high_]
    candidate = [low, value_low, slope_low]
    [new_high, new_value_high, new_slope_high] = optax.tree.where(
        set_high_to_low, candidate, default
    )

    # Set low to middle or keep as it is
    default = [low, value_low, slope_low]
    candidate = [middle, value_middle, slope_middle]
    [new_low, new_value_low, new_slope_low] = optax.tree.where(
        set_low_to_middle, candidate, default
    )

    # Update cubic reference point.
    # If high changed then it can be used as the new ref point.
    # Otherwise, low has been updated and not kept as high
    # so it can be used as the new ref point.
    [new_cubic_ref, new_value_cubic_ref] = optax.tree.where(
        set_high_to_middle | set_high_to_low,
        [high, value_high],
        [low, value_low],
    )
    # We stop if the searched interval is reduced below machine precision
    # and we already have found a positive stepsize ensuring sufficient
    # decrease. If no stepsize with sufficient decrease has been found,
    # we keep going on (some extremely steep functions require very small
    # stepsizes, see zakharov test in linesearch_test.py)
    max_iter_reached = (iter_num + 1) >= max_linesearch_steps
    presumably_failed = jnp.asarray(max_iter_reached) | (
        too_small_int & (new_safe_stepsize > 0.0)
    )
    failed = presumably_failed & ~done
    new_state = ZoomLinesearchState(
        count=numerics.safe_increment(iter_num),
        #
        params=params,
        updates=updates,
        stepsize_guess=state.stepsize_guess,
        #
        stepsize=middle,
        value=value_middle,
        grad=grad_middle,
        slope=slope_middle,
        #
        value_init=value_init,
        slope_init=slope_init,
        #
        decrease_error=decrease_error,
        curvature_error=curvature_error,
        error=new_error,
        #
        interval_found=state.interval_found,  # unchanged at this stage
        done=done,
        failed=failed,
        #
        low=new_low,
        value_low=new_value_low,
        slope_low=new_slope_low,
        high=new_high,
        value_high=new_value_high,
        slope_high=new_slope_high,
        cubic_ref=new_cubic_ref,
        value_cubic_ref=new_value_cubic_ref,
        #
        safe_stepsize=new_safe_stepsize,
        safe_value=new_safe_value,
        safe_grad=new_safe_grad,
    )
    return optax.tree.cast_like(new_state, other_tree=state)

  def _failure_diagnostic(state: ZoomLinesearchState) -> None:
    """Prints failure diagnostics."""
    jax.debug.print(FLAG_NO_STEPSIZE_FOUND)
    stepsize = state.stepsize

    slope_init = state.slope_init
    is_descent_dir = slope_init < 0.0
    _cond_print(
        ~is_descent_dir,
        FLAG_NOT_A_DESCENT_DIRECTION
        + "The slope (={slope_init}) at stepsize=0 should be negative",
        slope_init=slope_init,
    )
    _cond_print(
        is_descent_dir,
        "Consider augmenting the maximal number of linesearch iterations.",
    )
    eps = jnp.finfo(jnp.float32).eps
    below_eps = stepsize < eps
    _cond_print(
        below_eps & is_descent_dir,
        "Computed stepsize (={stepsize}) "
        "is below machine precision (={eps}), "
        "consider passing to higher precision like x64, using "
        "jax.config.update('jax_enable_x64', True).",
        stepsize=stepsize,
        eps=eps,
    )
    abs_slope_init = jnp.abs(slope_init)
    high_slope = abs_slope_init > 1e16
    _cond_print(
        high_slope & is_descent_dir,
        "Very large absolute slope at stepsize=0. "
        "(|slope|={abs_slope_init}). "
        "The objective is badly conditioned. "
        "Consider reparameterizing objective (e.g., normalizing parameters) "
        "or finding a better guess for the initial parameters for the "
        "solver.",
        abs_slope_init=abs_slope_init,
    )
    outside_domain = jnp.isinf(state.decrease_error)
    _cond_print(
        outside_domain,
        "Cannot even make a step without getting Inf or Nan. "
        + "The linesearch won't make a step and the optimizer is stuck.",
    )
    _cond_print(
        ~outside_domain,
        "Making an unsafe step, not decreasing enough the objective. "
        "Convergence of the solver is compromised as it does not reduce"
        " values.",
    )

  def init_fn(
      updates: base.Updates,
      params: base.Params,
      *,
      value: chex.Numeric,
      grad: base.Updates,
      prev_stepsize: chex.Numeric = 1.0,
      initial_guess_strategy: str = "one",
  ) -> ZoomLinesearchState:
    """Initializes the linesearch state."""

    if initial_guess_strategy == "one":
      stepsize_guess = jnp.asarray(1.0)
    elif initial_guess_strategy == "keep":
      stepsize_guess = prev_stepsize
    else:
      raise ValueError(
          f"Unknown initial guess strategy: {initial_guess_strategy}"
      )
    val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
    slope = optax.tree.real(optax.tree.vdot(updates, grad))
    return ZoomLinesearchState(
        count=jnp.asarray(0),
        #
        params=params,
        updates=updates,
        stepsize_guess=stepsize_guess,
        #
        stepsize=jnp.asarray(0.0, dtype=val_dtype),
        value=jnp.array(value, dtype=val_dtype),
        grad=grad,
        slope=slope,
        #
        value_init=value,
        slope_init=slope,
        #
        decrease_error=jnp.asarray(jnp.inf),
        curvature_error=jnp.asarray(jnp.inf),
        error=jnp.asarray(jnp.inf),
        #
        interval_found=jnp.asarray(False),
        done=jnp.asarray(False),
        failed=jnp.asarray(False),
        #
        low=jnp.asarray(0.0),
        value_low=value,
        slope_low=slope,
        high=jnp.asarray(0.0),
        value_high=value,
        slope_high=slope,
        cubic_ref=jnp.asarray(0.0),
        value_cubic_ref=value,
        #
        safe_stepsize=jnp.asarray(0.0),
        safe_value=value,
        safe_grad=grad,
    )

  def step_fn(
      state: ZoomLinesearchState,
      *,
      value_and_grad_fn: Callable[..., tuple[chex.Numeric, base.Updates]],
      fn_kwargs: dict[str, Any],
  ) -> ZoomLinesearchState:
    """Makes a step of the linesearch."""
    new_state = jax.lax.cond(
        state.interval_found,
        functools.partial(
            _zoom_into_interval,
            value_and_grad_fn=value_and_grad_fn,
            fn_kwargs=fn_kwargs,
        ),
        functools.partial(
            _search_interval,
            value_and_grad_fn=value_and_grad_fn,
            fn_kwargs=fn_kwargs,
        ),
        state,
    )
    new_state = jax.lax.cond(
        new_state.failed, _try_safe_step, lambda x: x, new_state
    )
    return optax.tree.cast_like(new_state, other_tree=state)

  def step_cond_fn(state: ZoomLinesearchState) -> Union[bool, chex.Numeric]:
    """Continuing criterion for the while loop of the linesearch."""
    return ~(state.done | state.failed)

  return init_fn, step_fn, step_cond_fn


class ZoomLinesearchInfo(NamedTuple):
  """Information about the zoom linesearch step, exposed for debugging.

  A positive curvature error is not stringent. It can be due to a maximal
  learning rate too small.
  A positive value in the sufficient curvature error is more problematic as it
  means that the algorithm may not be guaranteed to produce monotonically
  decreasing values.
  Consider using ``verbose=True`` in :func:`scale_by_zoom_linesearch` for
  additional failure diagnostics if the linesearch fails.

  Attributes:
    num_linesearch_steps: number of linesearch steps
    decrease_error: sufficient decrease error. A positive value indicates that
      the linesearch failed to find a stepsize that ensures a sufficient
      decrease. A null value indicates it succeeded in finding such a stepsize.
    curvature_error: small curvature error. A positive value indicates that the
      linesearch failed to find a stepsize that ensures a small curvature. A
      null value indicates it succeeded in finding such a stepsize.
  """

  num_linesearch_steps: Union[int, chex.Numeric]
  decrease_error: Union[float, chex.Numeric]
  curvature_error: Union[float, chex.Numeric]


class ScaleByZoomLinesearchState(NamedTuple):
  """State for scale_by_zoom_linesearch.

  Attributes:
    learning_rate: learning rate computed at the end of a round of line-search,
      used to scale the update.
    value: value of the objective computed at the end of a round of line-search.
      Can be reused using :func:`optax.value_and_grad_from_state`.
    grad: gradient of the objective computed at the end of a round of
      line-search. Can be reused using :func:`optax.value_and_grad_from_state`.
    info: Additional information on the status of the linesearch see
      :class:`otpax.ZoomLinesearchInfo`.
  """

  learning_rate: chex.Numeric
  value: chex.Numeric
  grad: base.Updates
  info: ZoomLinesearchInfo


def scale_by_zoom_linesearch(
    max_linesearch_steps: int,
    max_learning_rate: Optional[float] = None,
    tol: float = 0.0,
    increase_factor: float = 2.0,
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    approx_dec_rtol: Optional[float] = 1e-6,
    stepsize_precision: float = 1e-5,
    initial_guess_strategy: str = "keep",
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
  r"""Linesearch ensuring sufficient decrease and small curvature.

  This algorithm searches for a learning rate, a.k.a. stepsize, that satisfies
  both a sufficient decrease criterion, a.k.a. Armijo-Goldstein criterion,

  .. math::
    f(w + \eta u) \leq f(w) + \eta c_1 \langle u, \nabla f(w) \rangle + \epsilon
    \,,

  and a small curvature (along the update direction) criterion, a.k.a.
  Wolfe or second Wolfe criterion,

  .. math::
    |\langle \nabla f(w + \eta u), u \rangle| \leq c_2 |\langle \nabla f(w),
    \rangle| + \epsilon\,,

  where

  - :math:`f` is the function to minimize,
  - :math:`w` are the current parameters,
  - :math:`\eta` is the learning rate to find,
  - :math:`u` is the update direction,
  - :math:`c_1` is a coefficient (``slope_rtol``) measuring the relative
    decrease of the function in terms of the slope (scalar product between
    the gradient and the updates),
  - :math:`c_2` is a coefficient (``curv_rtol``) measuring the relative
    decrease of curvature.
  - :math:`\epsilon` is an absolute tolerance (``tol``).

  To deal with very flat functions, this linesearch switches from the sufficient
  decrease criterion presented above to an approximate sufficient decrease
  criterion introduced by Hager and Zhang (see [Hager and Zhang, 2006]).

  .. math::
    |\langle \nabla f(w+\eta u), u \rangle| \leq (2 c_1 - 1) |\langle \nabla
    f(w), \rangle| + \epsilon\,.

  The approximate curvature criterion is taken only if the values tried by the
  linesearch fall below a relative decrease of the initial function, that is,

  .. math::
    f(w + \eta u) \leq f(w) + c_3 |f(w)|

  where :math:`c_3` is a coefficient ``approx_dec_rtol`` measuring the relative
  decrease of the objective (see reference below and comments in the code for
  more details).

  The original sufficient decrease criterion can only capture
  differences up to :math:`\sqrt{\varepsilon_{machine}}` while the approximate
  sufficient decrease criterion can capture differences up to
  :math:`\varepsilon_{machine}` (see [Hager and Zhang, 2006]).
  Note that this add-on is not part of the original implementation of
  [Nocedal and Wright, 1999] and can be removed by
  setting ``approx_dec_rtol`` to ``None``.

  Args:
    max_linesearch_steps: maximum number of linesearch iterations.
    max_learning_rate: maximum admissible learning rate. Can be set to ``None``
      for no upper bound. A non ``None`` value may prevent the linesearch to
      find a learning rate satisfying the small curvature criterion, since the
      latter may require sufficiently large stepsizes.
    tol: tolerance on the criterions.
    increase_factor: increasing factor to augment the learning rate when
      searching for a valid interval containing a learning rate satisfying both
      criterions.
    slope_rtol: relative tolerance for the slope in the sufficient decrease
      criterion.
    curv_rtol: relative tolerance for the curvature in the small curvature
      criterion.
    approx_dec_rtol: relative tolerance for the initial value in the approximate
      sufficient decrease criterion. Can be set to ``None`` to use only the
      original Armijo-Goldstein decrease criterion.
    stepsize_precision: precision in the search of a stepsize satisfying both
      conditions. The algorithm proceeds with a bisection that refines an
      interval containing a stepsize satisfying both conditions. If that
      interval is reduced below ``stepsize_precision`` and a stepsize satisfying
      a sufficient decrease has been found, the algorithm selects that stepsize
      even if the curvature condition is not satisfied.
    initial_guess_strategy: initial guess for the learning rate used to start
      the linesearch. Can be either ``one`` or ``keep``. If ``one``, the initial
      guess is set to 1. If ``keep``, the initial guess is set to the learning
      rate of the previous step. We recommend to use ``keep`` if this linesearch
      is used in combination with SGD. We recommend to use ``one`` if this
      linesearch is used in combination with Newton methods or quasi-Newton
      methods such as L-BFGS.
    verbose: whether to print additional debugging information in case the
      linesearch fails.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs` object consisting in
    an init and an update function.

  Examples:
    An example on using the zoom line-search with SGD::

      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> solver = optax.chain(
      ...    optax.sgd(learning_rate=1.),
      ...    optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
      ... )
      >>> # Function with additional inputs other than params
      >>> def fn(params, x, y): return optax.l2_loss(x.dot(params), y)
      >>> params = jnp.array([1., 2., 3.])
      >>> opt_state = solver.init(params)
      >>> x, y = jnp.array([3., 2., 1.]), jnp.array(0.)
      >>> xs, ys = jnp.tile(x, (5, 1)), jnp.tile(y, (5,))
      >>> opt_state = solver.init(params)
      >>> print('Objective function: {:.2E}'.format(fn(params, x, y)))
      Objective function: 5.00E+01
      >>> for x, y in zip(xs, ys):
      ...   value, grad = jax.value_and_grad(fn)(params, x, y)
      ...   updates, opt_state = solver.update(
      ...       grad,
      ...       opt_state,
      ...       params,
      ...       value=value,
      ...       grad=grad,
      ...       value_fn=fn,
      ...       x=x,
      ...       y=y
      ...   )
      ...   params = optax.apply_updates(params, updates)
      ...   print('Objective function: {:.2E}'.format(fn(params, x, y)))
      Objective function: 2.56E-13
      Objective function: 2.84E-14
      Objective function: 0.00E+00
      Objective function: 0.00E+00
      Objective function: 0.00E+00

    A similar example, but with a non-stochastic function where we can reuse
    the value and the gradient computed at the end of the linesearch:

      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> # Function without extra arguments
      >>> def fn(params): return jnp.sum(params ** 2)
      >>> params = jnp.array([1., 2., 3.])
      >>> solver = optax.chain(
      ...    optax.sgd(learning_rate=1.),
      ...    optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
      ... )
      >>> opt_state = solver.init(params)
      >>> print('Objective function: {:.2E}'.format(fn(params)))
      Objective function: 1.40E+01
      >>> value_and_grad = optax.value_and_grad_from_state(fn)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...       grad, opt_state, params, value=value, grad=grad, value_fn=fn
      ...   )
      ...   params = optax.apply_updates(params, updates)
      ...   print('Objective function: {:.2E}'.format(fn(params)))
      Objective function: 0.00E+00
      Objective function: 0.00E+00
      Objective function: 0.00E+00
      Objective function: 0.00E+00
      Objective function: 0.00E+00

  References:
    Algorithms 3.5 3.6 of Nocedal and Wright, `Numerical Optimization
    <https://doi.org/10.1007/978-0-387-40065-5>`_, 1999

    Hager and Zhang `Algorithm 851: CG_DESCENT, a Conjugate Gradient Method
    with Guaranteed Descent
    <https://doi.org/10.1145/1132973.1132979>`_, 2006

  .. note::
    The curvature criterion can be avoided by setting by setting
    ``curv_rtol=jnp.inf``. The resulting algorithm will amount to a
    backtracking linesearch where a point satisfying sufficient decrease is
    searched by minimizing a quadratic or cubic approximation of the objective.
    This can be sufficient in practice and avoids having the linesearch spend
    many iterations trying to satisfy the small curvature criterion.

  .. note:: The algorithm can support complex inputs.

  .. seealso:: :func:`optax.value_and_grad_from_state` to make this method
    more efficient for non-stochastic objectives.
  """

  # Instantiates the linesearch with the given arguments.
  init_ls, step_ls, cond_step_ls = zoom_linesearch(
      max_linesearch_steps=max_linesearch_steps,
      max_stepsize=max_learning_rate,
      tol=tol,
      increase_factor=increase_factor,
      slope_rtol=slope_rtol,
      curv_rtol=curv_rtol,
      approx_dec_rtol=approx_dec_rtol,
      interval_threshold=stepsize_precision,
      verbose=verbose,
  )

  def init_fn(params: base.Params) -> ScaleByZoomLinesearchState:
    """Initializes state of scale_by_zoom_linesearch."""
    val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
    return ScaleByZoomLinesearchState(
        learning_rate=jnp.asarray(1.0, dtype=val_dtype),
        value=jnp.asarray(jnp.inf, dtype=val_dtype),
        grad=optax.tree.zeros_like(params),
        info=ZoomLinesearchInfo(
            num_linesearch_steps=jnp.asarray(0),
            decrease_error=jnp.asarray(jnp.inf),
            curvature_error=jnp.asarray(jnp.inf),
        ),
    )

  def update_fn(
      updates: base.Updates,
      state: ScaleByZoomLinesearchState,
      params: base.Params,
      *,
      value: chex.Numeric,
      grad: base.Updates,
      value_fn: Callable[..., tuple[chex.Numeric, base.Updates]],
      **extra_args: dict[str, Any],
  ) -> tuple[base.Updates, ScaleByZoomLinesearchState]:
    """Scales updates using the zoom linesearch.

    Args:
      updates: current updates.
      state: current state.
      params: current parameters.
      value: value of the function at the current params.
      grad: gradient of the function at the current params.
      value_fn: function returning the value of the function we seek to
        optimize.
      **extra_args: additional keyword arguments, if the function needs
        additional arguments such as input data, they should be put there.

    Returns:
      updates: updates for the params (new_params = params + updates).
      state: updated state.

    .. warning:: The objective to minimize, ``value_fn``, can take more than
        one input, but must return a single scalar (``float`` or scalar
        ``jax.Array``). If the function requires more than one input, the
        additional inputs need to be fed to the update, see the example in the
        docstring of the transform. The function value_fn needs to be amenable
        to differentiation in JAX.
    """
    # Fetch arguments to be fed to value_fn from the extra_args
    (fn_kwargs,), remaining_kwargs = utils._extract_fns_kwargs(  # pylint: disable=protected-access
        (value_fn,), extra_args
    )
    del remaining_kwargs
    value_and_grad_fn = jax.value_and_grad(value_fn)

    init_state = init_ls(
        updates,
        params,
        value=value,
        grad=grad,
        prev_stepsize=state.learning_rate,
        initial_guess_strategy=initial_guess_strategy,
    )

    final_state = jax.lax.while_loop(
        cond_step_ls,
        functools.partial(
            step_ls, value_and_grad_fn=value_and_grad_fn, fn_kwargs=fn_kwargs
        ),
        init_state,
    )
    learning_rate = final_state.stepsize
    scaled_updates = optax.tree.scale(learning_rate, updates)
    info_step = ZoomLinesearchInfo(
        num_linesearch_steps=final_state.count,
        decrease_error=final_state.decrease_error,
        curvature_error=final_state.curvature_error,
    )
    new_state = ScaleByZoomLinesearchState(
        learning_rate=learning_rate,
        value=final_state.value,
        grad=final_state.grad,
        info=info_step,
    )
    return scaled_updates, optax.tree.cast_like(new_state, other_tree=state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


# Flags to print errors, used for debugging, tested
FLAG_INTERVAL_NOT_FOUND = (
    "No interval satisfying curvature condition. "
    "Consider increasing maximal possible stepsize of the linesearch."
)
FLAG_INTERVAL_TOO_SMALL = (
    "Length of searched interval has been reduced below threshold."
)
FLAG_CURVATURE_COND_NOT_SATISFIED = (
    "Returning stepsize with sufficient decrease "
    "but curvature condition not satisfied."
)
FLAG_NO_STEPSIZE_FOUND = (
    "Linesearch failed, no stepsize satisfying sufficient decrease found."
)
FLAG_NOT_A_DESCENT_DIRECTION = (
    "The linesearch failed because the provided direction "
    "is not a descent direction. "
)


# =============================================================================
# Moré-Thuente Line Search Implementation
# =============================================================================

# Mathematical constants for the Moré-Thuente algorithm
def _get_machine_epsilon(dtype=jnp.float32):
  """Get machine epsilon for the given dtype."""
  return jnp.finfo(dtype).eps

XTRAPL, XTRAPU = 1.1, 4.0        # Extrapolation factors for bracket expansion
BISECTION_FACTOR = 0.66           # Threshold for forcing bisection

class MoreThuenteInfo(NamedTuple):
  """Information about the Moré-Thuente linesearch step, for debugging.

  Attributes:
    num_linesearch_steps: number of linesearch steps taken.
    decrease_error: error in the sufficient decrease criterion. A positive
      value indicates the criterion is violated.
    curvature_error: error in the strong curvature criterion. A positive
      value indicates the criterion is violated.
  """
  num_linesearch_steps: Union[int, chex.Numeric]
  decrease_error: Union[float, chex.Numeric]
  curvature_error: Union[float, chex.Numeric]

class ScaleByMoreThuenteState(NamedTuple):
  """State for :func:`optax.scale_by_more_thuente_linesearch`.

  Attributes:
    learning_rate: learning rate computed at the end of the line-search.
    value: value of the objective computed at the end of the line-search.
    grad: gradient of the objective computed at the end of the line-search.
    info: information about the Moré-Thuente linesearch step, for debugging.
  """
  learning_rate: chex.Numeric
  value: chex.Numeric
  grad: base.Updates
  info: MoreThuenteInfo

class IntervalState(NamedTuple):
  """Interval endpoints for line search bracketing."""
  stx: chex.Scalar    # Left endpoint step
  fx: chex.Scalar     # Function value at left endpoint
  gx: chex.Scalar     # Derivative at left endpoint
  sty: chex.Scalar    # Right endpoint step
  fy: chex.Scalar     # Function value at right endpoint
  gy: chex.Scalar     # Derivative at right endpoint

class LineSearchState(NamedTuple):
  """Internal state for Moré-Thuente line search algorithm."""
  count: chex.Scalar
  params: base.Params
  updates: base.Updates
  stp: chex.Scalar
  f: chex.Scalar
  g: chex.Scalar
  # Named tuple for better readability instead of packed array
  interval: IntervalState
  bounds: chex.Array    # [stmin, stmax, width, width1]
  search_params: chex.Array  # [finit, ginit, gtest, ftol, gtol, xtol]
  stage: chex.Scalar
  brackt: chex.Scalar
  nfev: chex.Scalar
  converged: chex.Scalar
  failed: chex.Scalar

@jax.jit
def _cubic_interpolation(step_a, func_a, grad_a, step_b, func_b, grad_b):
  """Compute cubic interpolation step between two points.

  Args:
    step_a, func_a, grad_a: step, function value, and derivative at point a
    step_b, func_b, grad_b: step, function value, and derivative at point b

  Returns:
    Cubic interpolation step
  """
  theta = 3 * (func_a - func_b) / (step_b - step_a) + grad_a + grad_b
  s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(grad_a)), jnp.abs(grad_b))
  epsmach = _get_machine_epsilon(s.dtype)
  s = jnp.maximum(s, epsmach)
  gamma = s * jnp.sqrt(jnp.maximum(0, (theta/s)**2 - (grad_a/s)*(grad_b/s)))
  gamma = jnp.where(step_b < step_a, -gamma, gamma)

  p = (gamma - grad_a) + theta
  q = ((gamma - grad_a) + gamma) + grad_b
  r = jnp.where(jnp.abs(q) > epsmach, p/q, 0)
  return step_a + r * (step_b - step_a)

@jax.jit
def _quadratic_interpolation(step_a, func_a, grad_a, step_b, func_b):
  """Compute quadratic interpolation step.

  Args:
    step_a, func_a, grad_a: step, function value, and derivative at point a
    step_b, func_b: step and function value at point b

  Returns:
    Quadratic interpolation step
  """
  denom = (func_a - func_b) / (step_b - step_a) + grad_a
  epsmach = _get_machine_epsilon(denom.dtype)
  return jnp.where(jnp.abs(denom) > epsmach,
                   step_a + grad_a / (2.0 * denom) * (step_b - step_a), step_a)

@jax.jit
def _secant_interpolation(step_a, grad_a, step_b, grad_b):
  """Compute secant interpolation step.

  Args:
    step_a, grad_a: step and derivative at point a
    step_b, grad_b: step and derivative at point b

  Returns:
    Secant interpolation step
  """
  grad_diff = grad_b - grad_a
  epsmach = _get_machine_epsilon(grad_diff.dtype)
  return jnp.where(jnp.abs(grad_diff) > epsmach,
                   step_b + grad_b / grad_diff * (step_a - step_b), step_b)

@jax.jit
def mcstep(interval, stp, fp, gp, bounds, brackt):
  """Compute new step using safeguarded cubic interpolation.

  This function is closely based on the FORTRAN implementation from:
  Moré, J. J. and Thuente, D. J. (1994). Line search algorithms with
  guaranteed sufficient decrease. ACM Transactions on Mathematical Software,
  20(3), 286-307.

  Original FORTRAN code: https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/

  Args:
    interval: IntervalState with interval endpoints.
    stp: current trial step size.
    fp: function value at trial step.
    gp: directional derivative at trial step.
    bounds: [stmin, stmax, width, width1] search bounds.
    brackt: whether interval brackets minimum.

  Returns:
    tuple (new_interval, new_step, new_brackt).
  """
  stmin, stmax = bounds[0], bounds[1]
  sgnd = gp * (interval.gx / jnp.abs(interval.gx))

  def case1():
    """Higher function value."""
    stc = _cubic_interpolation(interval.stx, interval.fx, interval.gx, stp, fp, gp)
    stq = _quadratic_interpolation(interval.stx, interval.fx, interval.gx, stp, fp)
    stf = jnp.where(jnp.abs(stc-interval.stx) < jnp.abs(stq-interval.stx),
                    stc, stc + (stq-stc)/2)
    return stf, 1

  def case2():
    """Opposite derivative signs."""
    stc = _cubic_interpolation(interval.stx, interval.fx, interval.gx, stp, fp, gp)
    sts = _secant_interpolation(interval.stx, interval.gx, stp, gp)
    stf = jnp.where(jnp.abs(stc-stp) > jnp.abs(sts-stp), stc, sts)
    return stf, 1

  def case3():
    """Decreasing derivative."""
    stc = _cubic_interpolation(interval.stx, interval.fx, interval.gx, stp, fp, gp)
    sts = _secant_interpolation(interval.stx, interval.gx, stp, gp)
    step = jnp.where(jnp.abs(stc-stp) < jnp.abs(sts-stp), stc, sts)

    conservative_factor = 0.5
    stf = jnp.where(brackt,
                    jnp.where(stp > interval.stx,
                             jnp.minimum(stp + conservative_factor*(interval.sty-stp),
                                        step),
                             jnp.maximum(stp + conservative_factor*(interval.sty-stp),
                                        step)),
                    jnp.clip(step, stmin, stmax))
    return stf, brackt

  def case4():
    """Increasing derivative."""
    cubic_step = _cubic_interpolation(interval.sty, interval.fy, interval.gy,
                                     stp, fp, gp)
    stf = jnp.where(brackt, cubic_step, jnp.where(stp > interval.stx, stmax, stmin))
    return stf, brackt

  # Select appropriate case based on function and derivative values
  # Use jax.lax.cond for lazy evaluation and cleaner nested conditions
  stf, new_brackt = jax.lax.cond(
      fp > interval.fx, case1,  # Higher function value
      lambda: jax.lax.cond(
          sgnd < 0, case2,  # Opposite derivative signs
          lambda: jax.lax.cond(
              jnp.abs(gp) < jnp.abs(interval.gx), case3,
              case4)))  # Decreasing vs increasing

  # Update interval endpoints based on the trial point results
  def update_interval():
    return jax.lax.cond(
        fp > interval.fx,
        lambda: IntervalState(interval.stx, interval.fx, interval.gx,
                              stp, fp, gp),  # Replace sty with trial point
        lambda: jax.lax.cond(
            sgnd < 0,
            lambda: IntervalState(stp, fp, gp, interval.stx, interval.fx,
                                  interval.gx),  # Replace stx with trial point
            lambda: IntervalState(stp, fp, gp, interval.sty, interval.fy,
                                  interval.gy)))  # Replace stx with trial point

  new_interval = update_interval()

  # Ensure the new step remains within bounds
  return new_interval, jnp.clip(stf, stmin, stmax), new_brackt

def more_thuente_linesearch(
    max_linesearch_steps: int = 20,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 0.1,
    verbose: bool = False
):
  """Create Moré-Thuente line search with strong Wolfe conditions.

  Args:
    max_linesearch_steps: maximum line search iterations.
    ftol: Armijo condition parameter in (0, 0.5).
    gtol: strong curvature parameter in (ftol, 1).
    xtol: relative width tolerance for convergence.
    verbose: print debugging information.

  Returns:
    tuple (init_fn, step_fn, cond_step_fn) for line search components.
  """

  def init_fn(updates, params, *, value, grad, stepsize_guess=1.0):
    """Initialize line search state."""
    derphi0 = optax.tree.real(optax.tree.vdot(updates, grad))

    return LineSearchState(
        count=jnp.array(0),
        params=params,
        updates=updates,
        stp=jnp.array(stepsize_guess),
        f=jnp.array(jnp.nan),
        g=jnp.array(jnp.nan),
        interval=IntervalState(0.0, value, derphi0, 0.0, value, derphi0),
        bounds=jnp.array([0.0, stepsize_guess * XTRAPU, 1e10, 1e10 / 0.5]),
        search_params=jnp.array([value, derphi0, ftol * derphi0, ftol, gtol, xtol]),
        stage=jnp.array(1),
        brackt=jnp.array(0),
        nfev=jnp.array(0),
        converged=jnp.array(0, dtype=jnp.int32),
        failed=jnp.array(0, dtype=jnp.int32))

  def step_fn(state, *, value_and_grad_fn, fn_kwargs):
    """Perform one Moré-Thuente line search iteration."""
    f_current, g_current = state.f, state.g
    finit, ginit, gtest, _, gtol_val, xtol_val = state.search_params
    stage, brackt = state.stage, state.brackt
    stmin, stmax, width, width1 = state.bounds

    # Check convergence conditions
    ftest = finit + state.stp * gtest
    armijo = f_current <= ftest
    curvature = jnp.abs(g_current) <= gtol_val * jnp.abs(ginit)
    converged = jnp.asarray(armijo & curvature, dtype=jnp.int32)

    current_width = jnp.abs(state.interval.sty - state.interval.stx)
    width_warning = brackt & (current_width <= xtol_val * stmax)
    bracket_warning = brackt & ((state.stp <= stmin) | (state.stp >= stmax))
    bounds_warning = ((state.stp >= 1e10) & (f_current <= ftest) &
                      (g_current <= gtest)) | \
                     ((state.stp <= 0.0) & ((f_current > ftest) |
                      (g_current >= gtest)))

    has_warning = bracket_warning | width_warning | bounds_warning
    failed = jnp.asarray((state.count >= max_linesearch_steps) | has_warning,
                         dtype=jnp.int32)
    converged = jnp.asarray(converged | width_warning, dtype=jnp.int32)
    should_stop = converged | failed

    def update_search():
      """Perform main line search update."""

      new_stage = jnp.where((stage == 1) & (f_current <= ftest) &
                             (g_current >= 0.0), 2, stage)
      use_modified = (stage == 1) & (f_current <= state.interval.fx) & \
                     (f_current > ftest)
      # The modified function is: ψ(α) = φ(α) - φ(0) - ftol*α*φ'(0)
      # This removes the linear trend to focus on curvature properties
      def apply_modified():
        return (
          f_current - state.stp * gtest,      # Modified function value
          g_current - gtest,                  # Modified derivative
          state.interval.fx - state.interval.stx * gtest,  # Modified fx
          state.interval.gx - gtest,          # Modified gx
          state.interval.fy - state.interval.sty * gtest,  # Modified fy
          state.interval.gy - gtest           # Modified gy
        )

      def use_original():
        return (f_current, g_current, state.interval.fx, state.interval.gx,
                state.interval.fy, state.interval.gy)

      fm, gm, fxm, gxm, fym, gym = jax.lax.cond(
          use_modified, apply_modified, use_original)

      modified_interval = IntervalState(
          state.interval.stx, fxm, gxm, state.interval.sty, fym, gym)
      new_interval, stp_new, new_brackt = mcstep(
          modified_interval, state.stp, fm, gm, state.bounds, brackt)

      def convert_back():
        return IntervalState(
            new_interval.stx,
            new_interval.fx + new_interval.stx * gtest,
            new_interval.gx + gtest,
            new_interval.sty,
            new_interval.fy + new_interval.sty * gtest,
            new_interval.gy + gtest
        )

      def use_original_back():
        return new_interval

      final_interval = jax.lax.cond(use_modified, convert_back, use_original_back)

      # Aggressive bisection for difficult optimization landscapes
      new_width = jnp.abs(final_interval.sty - final_interval.stx)
      force_bisection = new_brackt & ((new_width > 0.5 * width) |
                                       (new_width > 0.33 * width1))
      bisection_needed = new_brackt & (force_bisection |
                                        (new_width >= BISECTION_FACTOR * width1))
      stp_final = jnp.where(
          bisection_needed,
          final_interval.stx + 0.5 * (final_interval.sty - final_interval.stx),
          stp_new)

      conservative_extrap_lower = 1.05
      conservative_extrap_upper = 2.0

      new_stmin = jnp.where(new_brackt,
                           jnp.minimum(final_interval.stx, final_interval.sty),
                           jnp.where(~new_brackt,
                                    stp_final + conservative_extrap_lower *
                                    (stp_final - final_interval.stx),
                                    stmin))
      new_stmax = jnp.where(new_brackt,
                           jnp.maximum(final_interval.stx, final_interval.sty),
                           jnp.where(stp_final > final_interval.stx,
                                     stp_final + conservative_extrap_upper *
                                     (stp_final - final_interval.stx),
                                     stmax))

      return state._replace(
          count=state.count + 1,
          stp=jnp.clip(stp_final, 0.0, 1e10),
          f=f_current,
          g=g_current,
          interval=final_interval,
          bounds=jnp.array([new_stmin, new_stmax, new_width, width]),
          stage=new_stage,
          brackt=new_brackt,
          nfev=state.nfev + 1,
          converged=converged,
          failed=failed), jnp.asarray(should_stop, dtype=jnp.int32)

    # Use JAX conditional to either stop or continue the search
    return jax.lax.cond(
        should_stop,
        # If stopping, just update function values and status
        lambda: (state._replace(f=f_current, g=g_current, nfev=state.nfev + 1,
                               converged=converged, failed=failed),
                jnp.asarray(1, dtype=jnp.int32)),
        # Otherwise, perform the full search update
        update_search)

  def step_cond_fn(state):
    """Check if line search should continue."""
    return (state.converged | state.failed) == 0

  return init_fn, step_fn, step_cond_fn

def scale_by_more_thuente_linesearch(
    max_linesearch_steps: int = 20,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 0.1,
    initial_guess_strategy: str = "keep",
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
  r"""Moré-Thuente line search ensuring strong Wolfe conditions.

  Finds step size satisfying Armijo decrease and strong curvature conditions:
  f(x + α*p) ≤ f(x) + ftol*α*∇f(x)ᵀp and |∇f(x + α*p)ᵀp| ≤ gtol*|∇f(x)ᵀp|

  Args:
    max_linesearch_steps: Maximum line search iterations.
    ftol: Armijo condition parameter in (0, 0.5).
    gtol: Strong curvature parameter in (ftol, 1).
    xtol: Relative width tolerance for convergence.
    initial_guess_strategy: "keep" (reuse previous) or "one" (always 1.0).
    verbose: Print debugging info.

  Returns:
    GradientTransformationExtraArgs requiring value, grad, value_fn arguments.

  References:
    Moré & Thuente (1994). Line search algorithms with guaranteed sufficient decrease.
  """
  # Parameter validation
  if not 0 < ftol < 0.5:
    raise ValueError(f"ftol must be in (0, 0.5), got {ftol}")
  if not ftol < gtol < 1.0:
    raise ValueError(f"gtol must be in (ftol, 1), got gtol={gtol} with ftol={ftol}")
  if not 0 < xtol < 1.0:
    raise ValueError(f"xtol must be in (0, 1), got {xtol}")
  if max_linesearch_steps <= 0:
    raise ValueError(
        f"max_linesearch_steps must be positive, got {max_linesearch_steps}")
  if initial_guess_strategy not in ["keep", "one"]:
    raise ValueError(
        f"initial_guess_strategy must be 'keep' or 'one', "
        f"got '{initial_guess_strategy}'")

  init_ls, step_ls, cond_step_ls = more_thuente_linesearch(
      max_linesearch_steps, ftol, gtol, xtol, verbose)

  def init_fn(params):
    """Initialize optimization state."""
    val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
    return ScaleByMoreThuenteState(
        learning_rate=jnp.asarray(1.0, dtype=val_dtype),
        value=jnp.asarray(jnp.inf, dtype=val_dtype),
        grad=optax.tree.zeros_like(params),
        info=MoreThuenteInfo(
            num_linesearch_steps=jnp.asarray(0),
            decrease_error=jnp.asarray(jnp.inf),
            curvature_error=jnp.asarray(jnp.inf)
        )
    )

  def update_fn(updates, state, params, *, value, grad, value_fn, **extra_args):
    """Perform one optimization step using Moré-Thuente line search."""
    (fn_kwargs,), _ = utils._extract_fns_kwargs((value_fn,), extra_args)
    value_and_grad_fn = jax.value_and_grad(value_fn)

    stepsize_guess = (jnp.asarray(1.0) if initial_guess_strategy == "one"
                      else state.learning_rate)
    init_state = init_ls(updates, params, value=value, grad=grad,
                         stepsize_guess=stepsize_guess)

    def scan_fn(carry_state, _):
      search_state, should_stop = carry_state
      step_params = optax.tree.add_scale(search_state.params, search_state.stp,
                                          search_state.updates)
      f_trial, g_trial = value_and_grad_fn(step_params, **fn_kwargs)
      g_trial_dot = optax.tree.real(optax.tree.vdot(
          optax.tree.conj(g_trial), search_state.updates))

      new_state, new_should_stop = jax.lax.cond(
          should_stop,
          lambda: (search_state._replace(f=f_trial, g=g_trial_dot,
                                         nfev=search_state.nfev + 1 ),
                                         jnp.asarray(1, dtype=jnp.int32)),
          lambda: step_ls(
              search_state._replace(f=f_trial, g=g_trial_dot),
              value_and_grad_fn=value_and_grad_fn,
              fn_kwargs=fn_kwargs
          )
      )
      return (new_state, new_should_stop), None

    # Check if initial state already satisfies convergence conditions
    initial_converged = jnp.asarray(cond_step_ls(init_state) == 0, dtype=jnp.int32)

    # Run line search iterations using scan for efficiency
    (final_state, _), _ = jax.lax.scan(
        scan_fn, (init_state, initial_converged), None, length=max_linesearch_steps
    )

    learning_rate = final_state.stp
    finit, ginit, gtest, _, gtol_val, _ = final_state.search_params
    ftest = finit + final_state.stp * gtest
    decrease_error = jnp.maximum(0.0, final_state.f - ftest)
    curvature_error = jnp.maximum(0.0, jnp.abs(final_state.g) -
                                       gtol_val * jnp.abs(ginit))

    info = MoreThuenteInfo(
        num_linesearch_steps=final_state.count,
        decrease_error=decrease_error,
        curvature_error=curvature_error
    )

    final_params = optax.tree.add_scale(params, learning_rate, updates)
    final_value, final_grad = value_and_grad_fn(final_params, **fn_kwargs)

    new_state = ScaleByMoreThuenteState(
        learning_rate=learning_rate,
        value=final_value,
        grad=final_grad,
        info=info
    )

    return (optax.tree.scale(learning_rate, updates),
            optax.tree.cast_like(new_state, other_tree=state))

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
