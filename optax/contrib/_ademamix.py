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
from typing import NamedTuple, Optional, Tuple


def alpha_scheduler(alpha, alpha_start: float = 0, T_alpha: int = 0) -> base.Schedule:
  """The alpha scheduler from the paper.

  This is a progressive increase in alpha using a linear scheduler.

  Args:
    alpha: The current value of alpha (the coefficient that "blends" the two EMAs)
    alpha_start: The starting value of alpha
    T_alpha: The warmup time for alpha to reach it's final value.

  Returns:
    A `base.Schedule` object.

  """

  def schedule(step: int) -> float:
    is_warmup: float = (step < T_alpha).astype(jnp.float32)
    a: float = step / float(T_alpha)
    return is_warmup * ((1.0 - a) * alpha_start + a * alpha) + alpha * (1.0 - is_warmup)

  return schedule


def b3_scheduler(beta_end: float, beta_start: float = 0, T_b3: int = 0):
  """The b3 scheduler from the paper.

  This is a progressive increase in b3 attempting to increase t_half linearly
  (Appendix A.1 of the paper derives the scheduler.)

  Args:
    beta_end: The current value of b3 (the exponential decay rate to track the
      first moment of past gradients for the second EMA)
    beta_start: The starting value of b3
    T_b3: The warmup time for b3 to reach it's maximal value.

  Returns:
    A `base.Schedule` object.

  """

  def f(beta: float) -> float:
    return jnp.log(0.5) / jnp.log(beta) - 1

  def f_inv(t: float) -> float:
    return rsqrt(t + 1)

  def schedule(step: int) -> float:
    is_warmup = (step < T_b3).astype(jnp.float32)
    alpha = step / float(T_b3)
    return is_warmup * f_inv((1.0 - alpha) * f(beta_start) + alpha * f(beta_end)) + beta_end * (1.0 - is_warmup)

  return schedule


class ScaleByAdemamixState(NamedTuple):
   """State for the Ademamix algorithm."""

   count: chex.Array  # shape=(), dtype=jnp.int32.
   count_m2: chex.Array  # shape=(), dtype=jnp.int32.
   m1: base.Updates
   m2: base.Updates
   nu: base.Updates


def scale_by_ademamix(
  b1: float = 0.9,
  b2: float = 0.999,
  b3: base.ScalarOrSchedule = 0.9999,
  alpha: base.ScalarOrSchedule = 5.0,
  eps: float = 1e-8,
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

  Returns:
    A `GradientTransformation` object.

  Limitations: AdEMAMix consists in leveraging very old gradients. Therefore,
    the method is best suited to settings where the number of iterations is
    important. The paper reports on this effect in App. C.1.5, showing how
    smaller values of b3 (e.g. b3 = 0.999) can be better for low iterations
    scenarios. Moreover, retaining gradient information over many thousands
    steps can pose a problem in domains requiring fast adaptation to a sudden
    distribution shift, or general cases in which the distribution is
    non-stationary.
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
    c_b3 = b3_scheduler(state.count_m2) if callable(b3_scheduler) else b3
    c_alpha = (
      alpha_scheduler(state.count_m2) if callable(alpha_scheduler) else alpha
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
  weight_decay: float = 0.0,
) -> base.GradientTransformation:
  """The Ademamix optimizer.

  Description

  Examples:
    > import optax
    > import jax
    > import jax.numpy as jnp
    > def f(x): return jnp.sum(x ** 2)  # simple quadratic functio
    > solver = optax.ademamix(learning_rate=0.003)
    > params = jnp.array([1., 2., 3.])
    > print('Objective function: ', f(params))
      Objective function:  14.0
    > opt_state = solver.init(params
    > for _ in range(5):
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
    Pagliardini et al, 2024: https://arxiv.org/pdf/2409.03137

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
    weight_decay: Strength of the weight decay regularization.

  Returns:
    A `GradientTransformation` object.

  Limitations: AdEMAMix consists in leveraging very old gradients. Therefore,
    the method is best suited to settings where the number of iterations is
    important. The paper reports on this effect in App. C.1.5, showing how
    smaller values of b3 (e.g. b3 = 0.999) can be better for low iterations
    scenarios. Moreover, retaining gradient information over many thousands of
    steps can pose a problem in domains requiring fast adaptation to a sudden
    distribution shift, or general cases in which the distribution is
    non-stationary.
  """
  return combine.chain(
    scale_by_ademamix(b1, b2, b3, alpha, eps),
      transform.add_decayed_weights(weight_decay),
      transform.scale_by_learning_rate(learning_rate),
  )


if __name__ == "__main__":  # dummy test
  import jax
  import jax.numpy as jnp
  def f(x):
    return jnp.sum(x**2)  # simple quadratic function

  alpha = 8.0
  b1, b2, b3 = 0.9, 0.999, 0.9999

  f_a = alpha_scheduler(alpha, alpha_start=0, T_alpha=10)
  f_b3 = b3_scheduler(b3, beta_start=b1, T_b3=10)

  solver = ademamix(learning_rate=0.01, b1=b1, b2=b2, b3=f_b3, alpha=f_a, weight_decay=0.01)

  params = jnp.array([1.0, 2.0, 3.0])
  print("Objective function: {:.2f}".format(f(params)))
  opt_state = solver.init(params)
  for itr in range(100):
    grad = jax.grad(f)(params)
    updates, opt_state = solver.update(grad, opt_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    if itr % 5 == 0:
      print("Objective function: {:.2f}".format(f(params)))
  print(params)
