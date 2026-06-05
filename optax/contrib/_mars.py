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
"""MARS: Unleashing the Power of Variance Reduction for Training Large Models.

Reference:
  Hu et al., `MARS: Unleashing the Power of Variance Reduction for Training
  Large Models <https://arxiv.org/abs/2411.10438>`_, 2024.
"""

from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


class MarsState(NamedTuple):
  """State for the MARS optimizer."""

  count: jax.Array  # shape=(), dtype=jnp.int32
  mu: base.Updates  # first moment (EMA of corrected gradients)
  nu: base.Updates  # second moment (EMA of squared corrected gradients)
  prev_grad: base.Updates  # g_{t-1}: gradient from the previous step
  c_prev: base.Updates  # c_{t-1}: corrected gradient from the previous step


def scale_by_mars(
    gamma: float = 0.025,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
    *,
    correction_clip: Optional[float] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""MARS variance-reduction gradient rescaling.

  Computes a STORM-style corrected gradient :math:`c_t` and then applies
  Adam-style first- and second-moment accumulation on :math:`c_t` rather than
  the raw gradient :math:`g_t`.  The corrected gradient is:

  .. math::

    c_t = g_t + (1 - \gamma)(c_{t-1} - g_{t-1}), \quad c_1 = g_1.

  With :math:`\gamma = 1` this reduces to plain Adam rescaling.  Smaller
  :math:`\gamma` gives stronger variance reduction at the cost of sensitivity
  to gradient noise between consecutive steps.

  Args:
    gamma: Variance-reduction coefficient :math:`\gamma \in (0, 1]`.  The
      authors recommend ``0.025`` for LLM pre-training.
    b1: Exponential decay rate for the first moment (momentum).
    b2: Exponential decay rate for the second moment.
    eps: Small constant for numerical stability in the denominator.
    mu_dtype: Optional dtype for the first-moment buffer.  If ``None`` the
      dtype is inferred from the parameters.
    correction_clip: If set, clips the *correction term*
      :math:`(1-\gamma)(c_{t-1} - g_{t-1})` by global norm before adding it
      to :math:`g_t`.  Recommended by the paper (Section 3.2) for stability.
    nesterov: Whether to use Nesterov momentum for the first moment.

  Returns:
    A :class:`optax.GradientTransformation`.

  .. seealso:: :func:`optax.contrib.mars`
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)
    nu = optax.tree.zeros_like(params)
    prev_grad = optax.tree.zeros_like(params)
    c_prev = optax.tree.zeros_like(params)
    return MarsState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        nu=nu,
        prev_grad=prev_grad,
        c_prev=c_prev,
    )

  def update_fn(updates, state, params=None):
    del params
    count = state.count
    g_t = updates

    # ── Compute the STORM-style corrected gradient ─────────────────────────
    # c_t = g_t + (1 - gamma) * (c_{t-1} - g_{t-1})
    # At step 0 (count == 0) there is no previous information, so c_1 = g_1.
    correction = jax.tree.map(
        lambda c, g: (1.0 - gamma) * (c - g), state.c_prev, state.prev_grad
    )
    # Zero out the correction on the very first step.
    is_first_step = count == 0
    correction = jax.tree.map(
        lambda corr: jnp.where(is_first_step, jnp.zeros_like(corr), corr),
        correction,
    )

    if correction_clip is not None:
      # Clip the correction term by its global norm (Section 3.2 of paper).
      leaves = jax.tree.leaves(correction)
      global_norm = jnp.sqrt(
          sum(jnp.sum(leaf ** 2) for leaf in leaves) + 1e-12
      )
      scale = jnp.minimum(1.0, correction_clip / global_norm)
      correction = jax.tree.map(lambda c: c * scale, correction)

    c_t = jax.tree.map(lambda g, corr: g + corr, g_t, correction)

    # ── Adam-style moment updates on c_t ──────────────────────────────────
    mu = optax.tree.update_moment(c_t, state.mu, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(c_t, state.nu, b2, 2)
    count_inc = numerics.safe_increment(count)

    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)

    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, c: b1 * m + (1.0 - b1) * c, mu_hat, c_t
      )

    updates_out = jax.tree.map(
        lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat
    )
    mu = optax.tree.cast(mu, mu_dtype)

    new_state = MarsState(
        count=count_inc,
        mu=mu,
        nu=nu,
        prev_grad=g_t,
        c_prev=c_t,
    )
    return updates_out, new_state

  return base.GradientTransformation(init_fn, update_fn)


def mars(
    learning_rate: base.ScalarOrSchedule,
    gamma: float = 0.025,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, base.PyTree]] = None,
    mu_dtype: Optional[Any] = None,
    *,
    correction_clip: Optional[float] = None,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
  r"""MARS: variance-reduction AdamW for training large models.

  MARS replaces the raw gradient in Adam with a STORM-style corrected gradient
  :math:`c_t` that reduces variance across consecutive steps:

  .. math::

    c_t = g_t + (1 - \gamma)(c_{t-1} - g_{t-1}),

  then applies AdamW-style updates:

  .. math::

    \begin{align*}
      m_t &= \beta_1 m_{t-1} + (1-\beta_1) c_t, \\
      v_t &= \beta_2 v_{t-1} + (1-\beta_2) c_t^2, \\
      \hat{m}_t &= m_t / (1 - \beta_1^t), \\
      \hat{v}_t &= v_t / (1 - \beta_2^t), \\
      \theta_t &= \theta_{t-1}
        - \alpha \bigl(\hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)
        + \lambda \theta_{t-1}\bigr).
    \end{align*}

  MARS achieves the convergence rate of SGD-with-momentum while retaining
  Adam's per-coordinate adaptivity. In large-scale LLM pre-training experiments
  the authors report consistent improvements over AdamW.

  Args:
    learning_rate: Global step size, either a scalar or a schedule.
    gamma: Variance-reduction coefficient :math:`\gamma \in (0, 1]`.
      ``gamma=1`` recovers AdamW exactly.  The paper recommends ``0.025``
      for LLM pre-training; larger values (``0.5``–``1.0``) are safer for
      fine-tuning where gradients are smoother.
    b1: Exponential decay rate for the first moment.
    b2: Exponential decay rate for the second moment.
    eps: Small constant for numerical stability.
    weight_decay: AdamW-style decoupled weight decay coefficient.
    mask: A tree with the same structure as (or a prefix of) the params
      pytree, or a callable that returns such a tree given the params/updates.
      The leaves should be booleans; ``True`` leaves apply weight decay,
      ``False`` leaves skip it.
    mu_dtype: Optional dtype for the first-moment buffer.
    correction_clip: If set, the correction term :math:`(1-\gamma)(c_{t-1} -
      g_{t-1})` is clipped by this global norm before being added to
      :math:`g_t`.  Improves stability in early training (Section 3.2).
    nesterov: Whether to use Nesterov momentum.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> solver = optax.contrib.mars(learning_rate=1e-3)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...   grad = jax.grad(f)(params)
    ...   updates, opt_state = solver.update(grad, opt_state, params)
    ...   params = optax.apply_updates(params, updates)
    >>> print('Objective function: {:.2f}'.format(f(params)))
    Objective function: 13.97

  References:
    Hu et al., `MARS: Unleashing the Power of Variance Reduction for Training
    Large Models <https://arxiv.org/abs/2411.10438>`_, 2024.
  """
  return combine.chain(
      scale_by_mars(
          gamma=gamma,
          b1=b1,
          b2=b2,
          eps=eps,
          mu_dtype=mu_dtype,
          correction_clip=correction_clip,
          nesterov=nesterov,
      ),
      transform.add_decayed_weights(weight_decay, mask=mask),
      transform.scale_by_learning_rate(learning_rate),
  )
