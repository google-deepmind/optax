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
"""AdEMAMix.

Implementation of
"THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER"
(https://arxiv.org/pdf/2409.03137) by Matteo Pagliardini,
Pierre Ablin and David Grangier.
"""

from typing import Any, Callable, NamedTuple, Optional, Union
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


class ScaleByAdemamixState(NamedTuple):
  """State for the Ademamix algorithm.

  Attributes:
    count: iteration of the algorithm used to update the fast EMA and second
      moment.
    count_m2: iteration of the algorithm used to update the slow EMA and alpha.
    m1: fast EMA of the first moment
    m2: slow EMA of the first moment
    nu: estimate of the second moment
  """

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  count_m2: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  m1: base.Updates
  m2: base.Updates
  nu: base.Updates


def scale_by_ademamix(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    b3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 6.0,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformation:
  """Scale updates according to the Ademamix algorithm.

  See :func:`optax.contrib.ademamix.` for a full description of the algorithm.

  References:
    Pagliardini et al, `The AdEMAMix Optimizer: Better, Faster, Older
    <https://arxiv.org/abs/2409.03137>`_, 2024

  Args:
    b1: Exponential decay rate to track the fast EMA.
    b2: Exponential decay rate to track the second moment of past gradients.
    b3: Exponential decay rate to track the slow EMA.
    alpha: Mixing coefficient in the linear combination for the fast and slow
      EMAs.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params) -> ScaleByAdemamixState:
    m1 = optax.tree.zeros_like(params, dtype=mu_dtype)  # fast EMA
    m2 = optax.tree.zeros_like(params, dtype=mu_dtype)  # slow EMA
    nu = optax.tree.zeros_like(params, dtype=mu_dtype)  # second moment estimate
    return ScaleByAdemamixState(
        count=jnp.zeros([], jnp.int32),
        count_m2=jnp.zeros([], jnp.int32),
        m1=m1,
        m2=m2,
        nu=nu,
    )

  def update_fn(updates, state, params=None):
    del params
    c_b3 = b3(state.count_m2) if callable(b3) else b3
    c_alpha = alpha(state.count_m2) if callable(alpha) else alpha
    m1 = optax.tree.update_moment(
        updates, state.m1, b1, order=1
    )  # m1 = b1 * m1 + (1-b1) * updates
    m2 = optax.tree.update_moment(updates, state.m2, c_b3, order=1)
    nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, order=2)
    count_inc = numerics.safe_int32_increment(state.count)
    count_m2_inc = numerics.safe_int32_increment(state.count_m2)
    m1_hat = optax.tree.bias_correction(m1, b1, count_inc)
    # NOTE:  AdEMAMix does not perform bias correction on b2 to let
    # the slow EMA momentum buffer fill itself slowly.
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m1_, m2_, v_: (
            (m1_ + c_alpha * m2_) / (jnp.sqrt(v_ + eps_root) + eps)
        ),
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
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    b3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 5.0,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: jax.typing.ArrayLike = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
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
  :math:`S_0 := (m^{(1)}_0, m^{(2)}_0, \nu_0) = (0, 0, 0)`, representing initial
  estimates for the fast and slow EMAs of the first moment along with the second
  moment estimate. In practice, these values are stored as pytrees containing
  all zeros, with the same shape as the model updates.  At step :math:`t`,
  the ``update`` function of this optimizer takes as arguments the incoming
  gradients :math:`g^t`, the optimizer state :math:`S^t` and the parameters
  :math:`\theta^{(t)}`. It then computes updates :math:`\theta^{(t+1)}` and the
  new state :math:`S^{(t+1)}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_1^{(t)} &\leftarrow \beta_1 \cdot m_1^{(t-1)} + (1-\beta_1)
      \cdot g^{(t)} \\
      m_2^{(t)} &\leftarrow \beta_3 \cdot m_2^{(t-1)} + (1-\beta_3) \cdot
      g^{(t)} \\
      \nu^{(t)} &\leftarrow \beta_2 \cdot \nu^{(t-1)} + (1-\beta_2) \cdot
      {g^{(t)}}^2 \\
      \hat{m_1}^{(t)} &\leftarrow m_1^{(t)} / {(1-\beta_1^{(t)})} \\
      \hat{\nu}^{(t)} &\leftarrow \nu^{(t)} / {(1-\beta_2^{(t)})} \\
      \theta^{(t)} &\leftarrow \theta^{(t-1)} - \eta \cdot \left(
      \frac{(\hat{m_1}^{(t)} + \alpha m_2^{(t)})}{\left(\sqrt{\hat{\nu}^{(t)}
      + \bar{\varepsilon}} + \varepsilon\right)} + \lambda \theta^{(t-1)}
      \right).\\
      S^{(t)} &\leftarrow (m_1^{(t)}, m_2^{(t)}, v^{(t)}).
    \end{align*}

  .. note::
    AdEMAMix consists in leveraging very old gradients. Therefore,
    the method is best suited to settings where the number of iterations is
    important. The paper reports on this effect in Appendix C.1.5, showing how
    smaller values of ``b3`` (e.g. ``b3 = 0.999``) can be better for low
    iterations scenarios. Moreover, retaining gradient information over many
    thousands of steps can pose a problem in domains requiring fast adaptation
    to a sudden distribution shift, or general cases in which the distribution
    is non-stationary.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(jnp.square(x))  # simple quadratic function
    >>> solver = optax.contrib.ademamix(learning_rate=0.01)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.36E+01
    Objective function: 1.35E+01
    Objective function: 1.34E+01

  References:
    Pagliardini et al, `The AdEMAMix Optimizer: Better, Faster, Older
    <https://arxiv.org/abs/2409.03137>`_, 2024

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the fast EMA.
    b2: Exponential decay rate to track the second moment of past gradients.
    b3: Exponential decay rate to track the slow EMA.
    alpha: Mixing coefficient in the linear combination fo the fast and
      slow EMAs.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso::
    See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
    as the example
    :doc:`../../_collections/examples/contrib/rosenbrock_ademamix` for a use
    case.
  """
  return combine.chain(
      scale_by_ademamix(
          b1=b1,
          b2=b2,
          b3=b3,
          alpha=alpha,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


class ScaleBySimplifiedAdEMAMixState(NamedTuple):
  """State for the Simplified AdEMAMix optimizer.

  Attributes:
    t: iteration count
    m: EMA
    n: second moment estimate
  """

  t: jax.typing.ArrayLike
  m: base.Updates
  n: base.Updates


def lerp(t, a, b):
  return optax.tree.add_scale(a, t, optax.tree.sub(b, a))


def scale_by_simplified_ademamix(
    b1: jax.typing.ArrayLike = 0.99,
    b2: jax.typing.ArrayLike = 0.95,
    alpha: base.ScalarOrSchedule = 0.0,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
) -> base.GradientTransformation:
  """Scale updates according to the Simplified AdEMAMix optimizer.

  See :func:`optax.contrib.simplified_ademamix.` for a full description.

  References:
    Morwani et al, `Connections between Schedule-Free Optimizers, AdEMAMix, and
    Accelerated SGD Variants <https://arxiv.org/abs/2502.02431>`_, 2025

  Args:
    b1: Exponential decay rate to track the EMA.
    b2: Exponential decay rate to track the second moment of past gradients.
    alpha: Mixing coefficient for the current gradient and EMA.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.

  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params) -> ScaleBySimplifiedAdEMAMixState:
    return ScaleBySimplifiedAdEMAMixState(
        t=jnp.array(0, jnp.int32),
        m=optax.tree.zeros_like(params),
        n=optax.tree.zeros_like(params),
    )

  def update_fn(updates, state, params=None):
    del params
    g = updates
    m = optax.tree.add_scale(g, b1, state.m)
    n = lerp(b2, optax.tree.mul(g, g), state.n)

    t = numerics.safe_increment(state.t)

    n_hat = optax.tree.bias_correction(n, b2, t)

    u_num = optax.tree.add_scale(m, alpha, g)
    u_den = jax.tree.map(lambda n: jnp.sqrt(n + eps_root) + eps, n_hat)

    u = optax.tree.div(u_num, u_den)

    return u, ScaleBySimplifiedAdEMAMixState(t=t, m=m, n=n)

  return base.GradientTransformation(init_fn, update_fn)


def simplified_ademamix(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.99,
    b2: jax.typing.ArrayLike = 0.95,
    alpha: base.ScalarOrSchedule = 0.0,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    weight_decay: jax.typing.ArrayLike = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  r"""Simplified AdEMAMix.

  Simplified AdEMAMix (Adaptive EMA Mixture) is a simplified version of AdEMAMix that
  eliminates the need for maintaining two separate momentum buffers and removes the
  requirement for scheduling the mixing parameter :math:`\alpha`. Setting :math:`\alpha
  = 0` recovers the standard Adam optimizer, subject to appropriate transformations of
  :math:`\eta` and :math:`\beta_1`.

  Let :math:`\eta` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\alpha, \varepsilon, \bar{\varepsilon}`, represent the
  arguments ``b1``, ``b2``, ``alpha``, ``eps``  and ``eps_root``
  respectively. Let :math:`\lambda` be the weight decay and :math:`\theta_t`
  the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m^{(1)}_0, \nu_0) = (0, 0)`, representing initial
  estimates for the EMAs of the first moment along with the second
  moment estimate. In practice, these values are stored as pytrees containing
  all zeros, with the same shape as the model updates.  At step :math:`t`,
  the ``update`` function of this optimizer takes as arguments the incoming
  gradients :math:`g^t`, the optimizer state :math:`S^t` and the parameters
  :math:`\theta^{(t)}`. It then computes updates :math:`\theta^{(t+1)}` and the
  new state :math:`S^{(t+1)}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_1^{(t)} &\leftarrow \beta_1 \cdot m_1^{(t-1)} + g^{(t)} \\
      g^{(t)} \\
      \nu^{(t)} &\leftarrow \beta_2 \cdot \nu^{(t-1)} + (1-\beta_2) \cdot
      {g^{(t)}}^2 \\
      \hat{\nu}^{(t)} &\leftarrow \nu^{(t)} / {(1-\beta_2^{(t)})} \\
      \theta^{(t)} &\leftarrow \theta^{(t-1)} - \eta \cdot \left(
      \frac{(m_1^{(t)} + \alpha g^{(t)})}{\left(\sqrt{\hat{\nu}^{(t)}
      + \bar{\varepsilon}} + \varepsilon\right)} + \lambda \theta^{(t-1)}
      \right).\\
      S^{(t)} &\leftarrow (m_1^{(t)}, m_2^{(t)}, v^{(t)}).
    \end{align*}

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(jnp.square(x))  # simple quadratic function
    >>> solver = optax.contrib.simplified_ademamix(learning_rate=0.01)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.36E+01
    Objective function: 1.33E+01
    Objective function: 1.28E+01
    Objective function: 1.23E+01

  References:
    Morwani et al, `Connections between Schedule-Free Optimizers, AdEMAMix, and
    Accelerated SGD Variants <https://arxiv.org/abs/2502.02431>`_, 2025

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the EMA.
    b2: Exponential decay rate to track the second moment of past gradients.
    alpha: Mixing coefficient for the current gradient and EMA.
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
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.

  .. seealso::
    See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
    as the example
    :doc:`../../_collections/examples/contrib/rosenbrock_ademamix`.
  """  # noqa: E501
  return combine.chain(
      scale_by_simplified_ademamix(
          b1=b1,
          b2=b2,
          alpha=alpha,
          eps=eps,
          eps_root=eps_root,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )
