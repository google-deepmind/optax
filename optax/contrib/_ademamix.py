"""AdEMAMix.

Implementation of
"THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER"
(https://arxiv.org/pdf/2409.03137) by Matteo Pagliardini, 
Pierre Ablin and David Grangier.
"""

import chex
import jax.numpy as jnp
import jax.tree_util as jtu
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
import optax.tree_utils as otu
from jax.lax import rsqrt
from typing import NamedTuple, Tuple


def alpha_scheduler(
  alpha_value,
  alpha_start: float = 0,
  warmup_alpha: int = 0
) -> base.Schedule:
  """The alpha scheduler from the paper.

  This is a progressive increase in alpha using a linear scheduler.

  Args:
    alpha_value: The current value of alpha (the coefficient that "blends" 
      the two EMAs)
    alpha_start: The starting value of alpha
    warmup_alpha: The warmup time for alpha to reach it's final value.

  Returns:
    A `base.Schedule` object.

  """

  def schedule(step: int) -> float:
    is_warmup: float = jnp.array(step < warmup_alpha).astype(jnp.float32)
    a: float = step / float(warmup_alpha)
    return (
      is_warmup * ((1.0 - a) * alpha_start + a * alpha_value) +
      alpha_value * (1.0 - is_warmup)
    )
  return schedule


def b3_scheduler(
  beta_end: float,
  beta_start: float = 0,
  warmup_b3: int = 0
) -> base.Schedule:
  """The b3 scheduler from the paper.

  This is a progressive increase in b3 attempting to increase the number
  of iterations corresponding to where half of the mass of the second EMA
  is concentrated (denoted ``t_half`` in the paper).  This scheduler attempts
  to increase this value linearly.  Note for ``b3 = 0.9999, t_half`` is
  approximately ``6930.``
  (Appendix A.1 of the paper derives the scheduler.)

  Args:
    beta_end: The desired ending value of b3 (the exponential decay rate to 
      track the first moment of past gradients for the second EMA)
    beta_start: The starting value of b3
    warmup_b3: The warmup time for b3 to reach it's maximal value.

  Returns:
    A `base.Schedule` object.

  """

  def fun(beta: float) -> float:
    return jnp.log(0.5) / jnp.log(beta) - 1

  def f_inv(t: float) -> float:
    return rsqrt(t + 1)

  def schedule(step: int) -> float:
    is_warmup = jnp.array(step < warmup_b3).astype(jnp.float32)
    step_over_warmup = step / float(warmup_b3)
    return (
      is_warmup * f_inv((1.0 - step_over_warmup) * fun(beta_start)
      + step_over_warmup * fun(beta_end))
      + beta_end * (1.0 - is_warmup)
    )
  return schedule


class ScaleByAdemamixState(NamedTuple):
  """State for the Ademamix algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  count_m2: chex.Array  # shape=(), dtype=jnp.int32.
  m1: base.Updates
  m2: base.Updates
  nu: base.Updates


def scale_by_ademamix(
  b1: float,
  b2: float,
  b3: base.ScalarOrSchedule,
  alpha: base.ScalarOrSchedule,
  eps: float = 1e-8,
  eps_root: float = 0.0,
  weight_decay: float=0.0,
) -> base.GradientTransformation:
  """Rescale updates according to the Ademamix algorithm.

  References:
    [Pagliardini et al, 2024](https://arxiv.org/pdf/2409.03137)

  Args:
    b1: Exponential decay rate to track the first moment of past gradients for
      the first Exponential Moving Average (EMA) - same as AdamW
    b2: Exponential decay rate to track the second moment of past gradients for
      the first Exponential Moving Average (EMA) - same as AdamW
    b3: Exponential decay rate to track the first moment of past gradients
      for the second EMA.
    alpha: the coefficient that "blends" the two EMAs. paper states values in
      :math:`[4,10]` work well in practice.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: Term added to the denominator inside of the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    weight_decay: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    A `GradientTransformation` object.

  """

  def init_fn(params):
    m1 = otu.tree_zeros_like(params)  # fast EMA
    m2 = otu.tree_zeros_like(params)  # slow EMA
    nu = otu.tree_zeros_like(params)  # second moment estimate
    return ScaleByAdemamixState(
      count=jnp.zeros([], jnp.int32),
      count_m2=jnp.zeros([], jnp.int32),
      m1=m1,
      m2=m2,
      nu=nu,
    )

  def update_fn(
    updates: jtu.tree_map, state, params=None
  ) -> Tuple[jtu.tree_map, ScaleByAdemamixState]:
    del params
    c_b3 = b3_scheduler(state.count_m2) if callable(b3) else b3
    c_alpha = (
      alpha_scheduler(state.count_m2) if callable(alpha) else alpha
    )
    m1 = otu.tree_update_moment(
      updates, state.m1, b1, order=1
    )  # m1 = b1 * m1 + (1-b1) * updates
    m2 = otu.tree_update_moment(updates, state.m2, c_b3, order=1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, order=2)
    count_inc = numerics.safe_int32_increment(state.count)
    count_m2_inc = numerics.safe_int32_increment(state.count_m2)
    m1_hat = otu.tree_bias_correction(m1, b1, count_inc)
    # NOTE:  AdEMAMix does not perform bias correction on b2 to let the momentum
    # buffer fill itself slowly.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
      lambda m1_, m2_, v_: (m1_ + c_alpha * m2_) / (jnp.sqrt(v_) + eps),
      m1_hat,
      m2,
      nu_hat,
    )
    return updates, ScaleByAdemamixState(
      count=count_inc, count_m2=count_m2_inc, m1=m1, m2=m2, nu=nu
    )

  return base.GradientTransformation(init_fn, update_fn)


def ademamix(
  learning_rate: base.ScalarOrSchedule,
  b1: float = 0.9,
  b2: float = 0.999,
  b3: base.ScalarOrSchedule = 0.9999,
  alpha: base.ScalarOrSchedule = 5.0,
  eps: float = 1e-8,
  eps_root: float = 0.0,
  weight_decay: float = 0.0,
) -> base.GradientTransformation:
r"""AdEMAMix.

  AdEMAMix (Adaptive EMA Mixture) is AdamW with a mixture of two momentum 
  terms to better take advantage of historical gradients. 

  Both SGD with momemtum (SGD+M) and Adam incorporate momentum using
  Exponential Moving Averages (EMAs) of past gradients

  Let :math:`\eta` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\beta_3, \alpha, \varepsilon, \bar{\varepsilon}`, represent the 
  arguments ``b1``, ``b2``, ``b3``, ``alpha``, ``eps``  and ``eps_root``
  respectively. Let :math:`\lambda` be the weight decay and :math:`\theta_t` 
  the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m1_0, m2_0, v_0) = (0, 0, 0)`, representing initial estimates 
  for the first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t` 
  and the parameters :math:`\theta_t` and computes updates :math:`\theta_{t+1}` and 
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m1_t &\leftarrow \beta_1 \cdot m1_{t-1} + (1-\beta_1) \cdot g_t \\
      m2_t &\leftarrow \beta_3 \cdot m2_{t-1} + (1-\beta_3) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      \theta_t &\leftarrow \theta_{t-1} - \eta \cdot \left( (\hat{m1}_t +
      \alpha m2_t) / \left({\sqrt{\hat{v}_t + \bar{\varepsilon}} + \varepsilon\right) 
      + \lambda \theta_{t-1} \right).\\
      S_t &\leftarrow (m1_t, m2_t, v_t).
    \end{align*}

  Limitations: AdEMAMix consists in leveraging very old gradients. Therefore,
    the method is best suited to settings where the number of iterations is
    important. The paper reports on this effect in Appendix C.1.5, showing how
    smaller values of b3 (e.g. b3 = 0.999) can be better for low iterations
    scenarios. Moreover, retaining gradient information over many thousands of
    steps can pose a problem in domains requiring fast adaptation to a sudden
    distribution shift, or general cases in which the distribution is
    non-stationary.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamw(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    "THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER"
    (https://arxiv.org/pdf/2409.03137) by Matteo Pagliardini, 
    Pierre Ablin and David Grangier.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the fast EMA.
    b2: Exponential decay rate to track the second moment of past gradients.
    b3: Exponenital decay rate to track the slow EMA.
    alpha: Mixing coefficient in the linear combination fo the fast and slow EMAs. 
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso::
    See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
    as the example :doc:`../_collections/examples/contrib/rosenbrock_ademamix` for a use case.
  """
  return combine.chain(
    scale_by_ademamix(b1, b2, b3, alpha, eps, eps_root),
      transform.add_decayed_weights(weight_decay),
      transform.scale_by_learning_rate(learning_rate),
  )
