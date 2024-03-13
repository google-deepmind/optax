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

import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import utils
import optax.tree_utils as optax_tu


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
  """

  learning_rate: Union[float, jax.Array]
  value: Union[float, jax.Array]
  grad: Optional[base.Updates] = None


class BacktrackingSearchState(NamedTuple):
  """State during the inner loop of a backtracking line-search."""

  learning_rate: Union[float, jax.Array]
  new_value: Union[float, jax.Array]
  new_grad: base.Updates
  accepted: bool
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
) -> base.GradientTransformationExtraArgs:
  r"""Backtracking line-search ensuring sufficient decrease (Armijo criterion).

  Selects learning rate :math:`\gamma` such that it verifies the decrease
  condition

  .. math::
      f(w + \gamma u) \leq (1+\delta)f(w)
        + \gamma c \langle u, \nabla f(w) \rangle + \epsilon \,,

  where :math:`f` is the function to minimize, :math:`\gamma` is the learning
  rate to find, :math:`u` is the update direction, :math:`c` is a coefficient
  (``slope_rtol``) measuring the relative decrease of the function in terms of
  the slope (scalar product between the gradient and the updates),
  :math:`\delta` is a relative tolerance (``rtol``), and :math:`\epsilon` is
  an absolute tolerance (``atol``).

  The algorithm starts with a given guess of a learning rate and decrease it
  by ``decrease_factor`` until the criterion above is met.

  .. warning::
    The sufficient decrease condition might be impossible to satisfy for some
    update directions. To guarantee a non-trivial solution for the sufficient
    decrease condition, employ a descent direction for updates (:math:`u`). An
    update (:math:`u`) is considered a descent direction if the derivative of
    :math:`f(w + \gamma u)` at :math:`\gamma = 0`
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

  .. seealso:: :func:`optax.value_and_grad_from_state` to make this method
    more efficient for non-stochastic objectives.

  .. versionadded:: 0.2.0

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
    atol: absolute tolerance at which the condition needs to be satisfied.
    rtol: relative tolerance at which the condition needs to be satisfied.
    store_grad: whether to compute and store the gradient at the end of the
      linesearch. Since the function is called to compute the value to accept
      the learning rate, we can also access the gradient along the way. By doing
      that, we can directly reuse the value and the gradient computed at the end
      of the linesearch for the next iteration using
      :func:`optax.value_and_grad_from_state`. See the example above.

  Returns:
    A :class:`GradientTransformationExtraArgs`, where the ``update`` function
    takes the following additional keyword arguments:
      * value: value of the function at the current params.
      * grad: gradient of the function at the current params.
      * value_fn: function returning the value of the function we seek to
        optimize.
      * **extra_args: additional keyword arguments, if the function needs
          additional arguments such as input data, they should be put there (
          see example in this docstrihng).
  """

  def init_fn(params: base.Params) -> ScaleByBacktrackingLinesearchState:
    if store_grad:
      grad = optax_tu.tree_zeros_like(params)
    else:
      grad = None
    return ScaleByBacktrackingLinesearchState(
        learning_rate=jnp.array(1.0),
        value=jnp.array(jnp.inf),
        grad=grad,
    )

  def _check_condition(learning_rate, slope, value, new_value):
    violation = (
        new_value - (1 + rtol) * value - learning_rate * slope_rtol * slope
    )
    violation = jnp.where(jnp.isnan(violation), jnp.inf, violation)
    return violation <= atol

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

    .. warning:: The objective to minimize, ``value_fn``, can take more than
        one input, but must return a single scalar (float or jax.Array of
        dimension one). If the function requires more than one input, the
        additional inputs need to be fed to the update, see the example in the
        docstring of the transform. The function value_fn needs to be amenable
        to differentiation in JAX.

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
    """
    # Fetch arguments to be fed to value_fn from the extra_args
    (fn_kwargs,), remaining_kwargs = utils._extract_fns_kwargs(  # pylint: disable=protected-access
        (value_fn,), extra_args
    )
    del remaining_kwargs

    # Slope of lr -> value_fn(params + lr * updates) at lr = 0
    # Should be negative to ensure that there exists a lr (potentially
    # infinitesimal) that satisfies the criterion.
    slope = optax_tu.tree_vdot(updates, grad)

    def cond_fn(
        search_state: BacktrackingSearchState,
    ) -> Union[int, jax._src.basearray.Array]:
      """Whether to stop the line-search inner loop."""
      accepted = search_state.accepted
      iter_num = search_state.iter_num
      return (~accepted) & (iter_num <= max_backtracking_steps)

    def body_fn(
        search_state: BacktrackingSearchState,
    ) -> BacktrackingSearchState:
      """Line-search inner loop step."""
      learning_rate = search_state.learning_rate
      new_grad = search_state.new_grad
      iter_num = search_state.iter_num
      # We start decreasing the learning rate after the first iteration
      # and up until the criterion is accepted.
      learning_rate = jnp.where(
          iter_num > 0, decrease_factor * learning_rate, learning_rate
      )
      new_params = optax_tu.tree_add_scalar_mul(params, learning_rate, updates)

      value_fn_ = functools.partial(value_fn, **fn_kwargs)
      if store_grad:
        # We evaluate value_fn and get its jvp operator so that we can
        # compute the gradient by transposing the jvp.
        new_value, jvp_value_fn = jax.linearize(value_fn_, new_params)

        accepted = _check_condition(learning_rate, slope, value, new_value)
        # If the line-search ends, we get the gradient for the new round of
        # line-search.
        new_grad = jax.lax.cond(
            accepted | (iter_num == max_backtracking_steps),
            lambda p: jax.linear_transpose(jvp_value_fn, p)(1.0)[0],
            lambda *_: new_grad,
            new_params,
        )
      else:
        # Here we just compute the value and leave the gradient as is
        new_value = value_fn_(new_params)
        accepted = _check_condition(learning_rate, slope, value, new_value)
      search_state = BacktrackingSearchState(
          learning_rate=learning_rate,
          new_value=new_value,
          new_grad=new_grad,
          accepted=accepted,
          iter_num=iter_num + 1,
      )
      return search_state

    # We start with a guess candidate learning rate that may be larger than
    # the current one but no larger than the maximum one.
    learning_rate = jnp.minimum(
        increase_factor * state.learning_rate, max_learning_rate
    )

    search_state = BacktrackingSearchState(
        learning_rate=learning_rate,
        new_value=value,
        new_grad=optax_tu.tree_zeros_like(params),
        accepted=False,
        iter_num=0,
    )
    search_state = jax.lax.while_loop(cond_fn, body_fn, search_state)

    # If store_grad is False we simply return None (to not mix up with
    # optax_tu.tree_zeros_like(params))
    new_grad = search_state.new_grad if store_grad else None
    new_value = search_state.new_value
    new_learning_rate = search_state.learning_rate

    # At the end, we just scale the updates with the learning rate found.
    new_updates = optax_tu.tree_scalar_mul(new_learning_rate, updates)

    new_state = ScaleByBacktrackingLinesearchState(
        learning_rate=new_learning_rate,
        value=new_value,
        grad=new_grad,
    )
    return new_updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
