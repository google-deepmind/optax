"""AdeMAMix.

Implementation of
"THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER"
(https://arxiv.org/pdf/2409.03137) by Matteo Pagliardini, 
Pierre Ablin and David Grangier.
"""
import optax.tree_utils as otu
from typing import NamedTuple, Optional
import chex
import jax.numpy as jnp
import jax.tree_util as jtu
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform


class ScaleByAdemamixState(NamedTuple):
  """State for the Ademamix algorithm."""

  count: chex.Array
  count_m2: chex.Array
  m1: base.Updates
  m2: base.Updates
  nu: base.Updates


def scale_by_ademamix(
  b1: float = 0.9,
  b2: float = 0.999,
  b3: float = 0.9999,
  alpha: float = 5.0,
  b3_scheduler: Optional[base.ScalarOrSchedule] = None,
  alpha_scheduler: Optional[base.ScalarOrSchedule] = None,
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
    b3_scheduler: The schedule for the b3 parameter
    alpha_scheduler: The schedule for the alpha parameter
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

  def update_fn(updates, state, params=None):
    del params
    c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
    c_alpha = (
      alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
    )
    m1 = otu.tree_update_moment(
      updates, state.m1, b1, 1
    )  # m1 = b1 * m1 + (1-b1) * updates
    m2 = otu.tree_update_moment(updates, state.m2, c_b3, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    count_m2_inc = numerics.safe_int32_increment(state.count_m2)
    m1_hat = otu.tree_bias_correction(m1, b1, count_inc)
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
  b3: float = 0.9999,
  alpha: float = 5.0,
  b3_scheduler: Optional[base.ScalarOrSchedule] = None,
  alpha_scheduler: Optional[base.ScalarOrSchedule] = None,
  eps: float = 1e-8,
  weight_decay: float = 0.0,
) -> base.GradientTransformation:
  """The Ademamix optimiser.

  Description

  Examples:
    > import optax
    > import jax
    > import jax.numpy as jnp
    > def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    > solver = optax.ademamix(learning_rate=0.003)
    > params = jnp.array([1., 2., 3.])
    > print('Objective function: ', f(params))
      Objective function:  14.0
    > opt_state = solver.init(params)
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
    b3_scheduler: The schedule for the b3 parameter
    alpha_scheduler: The schedule for the alpha parameter
    eps: A small constant applied to denominator outside of the square root
         (as in the Adam paper) to avoid dividing by zero when rescaling.
    weight_decay: Strength of the weight decay regularization.

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
  return combine.chain(
    scale_by_ademamix(
      b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps
    ),
    transform.add_decayed_weights(weight_decay),
    transform.scale_by_learning_rate(learning_rate),
  )
