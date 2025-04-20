from typing import Optional, Any
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src import combine
from optax._src import transform
import optax.tree_utils as otu


def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
) -> base.GradientTransformation:
  r"""Rescale updates according to the ADOPT algorithm.

  ADOPT (Modified Adam Can Converge with Any β2 with the Optimal Rate) is a variant
  of Adam that can converge with any β2 value while maintaining the optimal rate.
  
  This implementation includes a clipping operation to improve stability, especially
  in the early stages of training. The clipping helps avoid near-zero divisions when
  some elements of the parameter gradient are near zero at initialization.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.
    use_clipping: Whether to use gradient clipping to improve stability.
      When enabled, the clipping value is set to step**0.25, which aligns
      with the theory to ensure convergence.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    b2_ = jnp.where(state.count > 0, b2, 0)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2_, 2)
    if use_clipping:
      clip_value = state.count ** 0.25
      mu_updates = jax.tree.map(lambda ud, nu: jnp.clip(ud / jnp.maximum(jnp.sqrt(nu), eps), -clip_value, clip_value), updates, state.nu)
      b1_ = b1
    else:
      mu_updates = jax.tree.map(lambda ud, nu: ud / jnp.maximum(jnp.sqrt(nu), eps), updates, state.nu)
      b1_ = jnp.where(state.count > 0, b1, 0)
    mu = otu.tree_update_moment(mu_updates, state.mu, b1_, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_ = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          mu,
          mu_updates,
      )
    else:
      mu_ = mu  
    updates = jnp.where(state.count > 0, mu_, otu.tree_zeros_like(mu_, dtype=mu_dtype))
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def adopt(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
) -> base.GradientTransformationExtraArgs:
  r"""ADOPT: Modified Adam optimizer that can converge with any beta2 value.

  ADOPT (Adaptive Optimization with Provable Theoretical guarantees) is a modified 
  version of Adam that ensures convergence with any β2 value while maintaining the 
  optimal convergence rate. This implementation includes an optional clipping 
  operation to improve stability, especially in early training stages.

  The key difference from Adam is that ADOPT modifies the update rule to avoid 
  potential instability issues, particularly when some gradient elements are near 
  zero at initialization. This can happen when parameters (e.g., the last layer of 
  a neural network) are initialized with zeros.

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. At step :math:`t`, the ``update`` function computes:

  With clipping enabled (default), ADOPT applies a clipping operation to improve 
  stability, particularly in early training stages.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root,
      to avoid dividing by zero when rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.
    use_clipping: Whether to apply clipping to improve stability. Recommended
      to keep as True, especially for training from scratch.

  Returns:
    The corresponding :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adopt(learning_rate=0.003)
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
    Taniguchi et al, `ADOPT: Modified Adam Can Converge with Any β2 with the 
    Optimal Rate <https://arxiv.org/abs/2403.00855>`_, NeurIPS 2024
  """
  return combine.chain(
      transform.scale_by_adopt(
          b1=b1,
          b2=b2,
          eps=eps,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
          use_clipping=use_clipping,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )