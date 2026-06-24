# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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

Implementation of "MARS: Unleashing the Power of Variance Reduction for
Training Large Models" (https://arxiv.org/abs/2411.10438) by Huizhuo Yuan,
Yifeng Liu, Shuang Wu, Xun Zhou, and Quanquan Gu.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils
from optax.transforms import _adding
from optax._src import transform
import optax.tree


class ScaleByMARSState(NamedTuple):
  """State for the MARS variance-reduction preconditioner."""

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32
  last_grad: base.Updates  # gradient from the previous step, in float32
  mu: base.Updates         # first-moment estimate (projected into EMA)
  nu: base.Updates         # second-moment estimate


def scale_by_mars(
    b1: jax.typing.ArrayLike = 0.95,
    b2: jax.typing.ArrayLike = 0.99,
    eps: jax.typing.ArrayLike = 1e-8,
    gamma: jax.typing.ArrayLike = 0.025,
    clip_threshold: Optional[float] = 1.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""Scale updates using the MARS variance-reduction preconditioner.

  See :func:`optax.contrib.mars` for full details.

  Args:
    b1: Decay rate for the first moment (momentum) estimates.
    b2: Decay rate for the second moment estimates.
    eps: Small constant added to the denominator for numerical stability.
    gamma: Variance-reduction mixing coefficient. Controls how strongly the
      optimizer corrects for the change in gradient direction since the last
      step. Setting ``gamma=0`` recovers standard Adam.
    clip_threshold: If set, the variance-reduced gradient :math:`c_t` is
      rescaled to have at most this L2 norm before the moment updates. This
      prevents large corrections early in training (when ``last_grad`` is
      near zero) from dominating the update. Set to ``None`` to disable.
    mu_dtype: Optional dtype for the first moment buffer. Useful for reducing
      memory in mixed-precision training. If ``None``, inferred from params.

  Returns:
    A :class:`optax.GradientTransformation`.
  """
  # Normalize to float32 so Python-float and JAX-float32 closures compute
  # (1 - b2) identically. Without this, the Python path gets 1-0.99=0.01
  # while the JAX float32 path gets 1-float32(0.99)=float32(0.009999...).
  # inject_hyperparams always passes strongly-typed float32 values, so this
  # cast ensures direct and inject paths are numerically identical.
  b1 = jnp.asarray(b1, dtype=jnp.float32)
  b2 = jnp.asarray(b2, dtype=jnp.float32)
  eps = jnp.asarray(eps, dtype=jnp.float32)
  gamma = jnp.asarray(gamma, dtype=jnp.float32)

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params: base.Params) -> ScaleByMARSState:
    param_leaves = jax.tree.leaves(params)
    if not param_leaves:
      # Empty-params guard for tree_map_params compatibility.
      empty = params
      return ScaleByMARSState(
          count=jnp.zeros([], jnp.int32),
          last_grad=empty,
          mu=empty,
          nu=empty,
      )
    return ScaleByMARSState(
        count=jnp.zeros([], jnp.int32),
        last_grad=optax.tree.zeros_like(params),
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        nu=optax.tree.zeros_like(params),
    )

  def update_fn(
      updates: base.Updates,
      state: ScaleByMARSState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, ScaleByMARSState]:
    del params
    count_inc = numerics.safe_int32_increment(state.count)

    # Variance-reduced gradient:
    #   c_t = g_t + γ · (β₁/(1-β₁)) · (g_t - g_{t-1})
    # This is the approximate-MARS variant: g_{t-1} is the gradient from the
    # previous step on a different mini-batch, not a re-evaluation on the same
    # batch. The approximation is cheaper (one grad eval per step) and works
    # well in practice. When γ=0, c_t = g_t and MARS is identical to Adam.
    correction_scale = gamma * b1 / (1.0 - b1)
    c = jax.tree.map(
        lambda g, g_prev: (
            g.astype(jnp.float32)
            + correction_scale
            * (g.astype(jnp.float32) - g_prev.astype(jnp.float32))
        ),
        updates,
        state.last_grad,
    )

    # Clip the corrected gradient to at most `clip_threshold` in L2 norm.
    # Without clipping, the very first step (when last_grad=0) amplifies the
    # gradient by a factor of (1 + γβ₁/(1-β₁)), which can be ~1.5× for
    # typical γ=0.025, β₁=0.95. For larger γ the amplification is stronger;
    # clipping keeps training numerically stable regardless of γ.
    if clip_threshold is not None:
      c_norm = optax.tree.norm(c)
      clip_scale = jnp.minimum(
          jnp.ones([], dtype=jnp.float32),
          jnp.asarray(clip_threshold, dtype=jnp.float32) / (c_norm + 1e-12),
      )
      c = jax.tree.map(lambda ci: ci * clip_scale, c)

    mu_new = jax.tree.map(
        lambda m, ci: b1 * m.astype(jnp.float32) + (1.0 - b1) * ci,
        state.mu,
        c,
    )
    nu_new = jax.tree.map(
        lambda v, ci: b2 * v.astype(jnp.float32) + (1.0 - b2) * jnp.square(ci),
        state.nu,
        c,
    )

    mu_hat = jax.tree.map(lambda m: m / (1.0 - b1**count_inc), mu_new)
    nu_hat = jax.tree.map(lambda v: v / (1.0 - b2**count_inc), nu_new)

    new_updates = jax.tree.map(
        lambda m, v, g: (m / (jnp.sqrt(v) + eps)).astype(g.dtype),
        mu_hat,
        nu_hat,
        updates,
    )

    # Cast moments back to their stored dtypes so dtype is stable across steps.
    # Using the stored tensor's dtype (not mu_dtype directly) handles the
    # mu_dtype=None case, where mu was initialised with the param dtype.
    mu_stored = jax.tree.map(
        lambda m_new, m: m_new.astype(m.dtype), mu_new, state.mu
    )
    nu_stored = jax.tree.map(
        lambda v_new, v: v_new.astype(v.dtype), nu_new, state.nu
    )

    return new_updates, ScaleByMARSState(
        count=count_inc,
        # Store last_grad in the same dtype as the incoming gradient; the
        # float32 promotion happens during the correction computation above.
        last_grad=updates,
        mu=mu_stored,
        nu=nu_stored,
    )

  # pyrefly: ignore[bad-argument-type]
  return base.GradientTransformation(init_fn, update_fn)


def mars(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.95,
    b2: jax.typing.ArrayLike = 0.99,
    eps: jax.typing.ArrayLike = 1e-8,
    gamma: jax.typing.ArrayLike = 0.025,
    clip_threshold: Optional[float] = 1.0,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""MARS: variance-reduced Adam for large-model training.

  MARS (arXiv:2411.10438) augments Adam with a scaled stochastic recursive
  momentum correction that reduces gradient variance across steps. For each
  parameter, the raw gradient :math:`g_t` is replaced by a variance-reduced
  estimate :math:`c_t` before the Adam moment updates:

  .. math::

    \begin{align*}
      c_t &\leftarrow g_t + \gamma \frac{\beta_1}{1 - \beta_1}
            (g_t - g_{t-1}) \\
      \tilde{c}_t &\leftarrow c_t \,/\, \max\!\bigl(1,\, \|c_t\|_2\bigr) \\
      m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \tilde{c}_t \\
      v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \tilde{c}_t^2 \\
      \hat{m}_t &\leftarrow m_t / (1 - \beta_1^t) \\
      \hat{v}_t &\leftarrow v_t / (1 - \beta_2^t) \\
      \Delta_t &\leftarrow \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)
    \end{align*}

  This is the *approximate* MARS variant: :math:`g_{t-1}` is the gradient
  from the previous step rather than a re-evaluation on the same mini-batch.
  This costs one gradient evaluation per step (identical to Adam) while still
  capturing most of the variance-reduction benefit.

  Setting :math:`\gamma = 0` recovers standard Adam exactly.

  .. note::
    MARS adds a ``last_grad`` buffer to the optimizer state, increasing memory
    by one copy of the parameter tensors beyond standard Adam. For large models
    this is the same overhead as adding a second optimizer slot (e.g. the
    second moment in Adam).

  .. note::
    The paper reports best results on transformer language model training with
    ``learning_rate=3e-3``, ``b1=0.95``, ``b2=0.99``, ``gamma=0.025``, and
    ``weight_decay=0.1``. These differ noticeably from typical AdamW defaults;
    hyperparameter transfer from AdamW is not straightforward.

  Args:
    learning_rate: A global scaling factor, either fixed or a schedule; see
      :func:`optax.scale_by_learning_rate`.
    b1: Decay rate for the first moment (momentum) estimates.
    b2: Decay rate for the second moment estimates.
    eps: Small constant added to the denominator for numerical stability.
    gamma: Variance-reduction mixing coefficient. Controls how strongly the
      optimizer corrects for gradient direction changes between steps. The
      paper uses ``gamma=0.025``; larger values give more aggressive variance
      reduction but can destabilize training if gradients are noisy.
    clip_threshold: Maximum L2 norm of the variance-reduced gradient
      :math:`c_t` before moment updates. Prevents the large effective
      gradient on the first step (when ``last_grad`` is zero) from causing
      an outsized update. Defaults to ``1.0``; set to ``None`` to disable.
    weight_decay: Optional :math:`\ell_2` regularization strength.
    weight_decay_mask: A tree with the same structure as (or a prefix of) the
      params pytree, or a callable that returns such a tree given the params.
      Leaves should be booleans indicating which parameters to apply weight
      decay to.
    mu_dtype: Optional dtype for the first moment buffer; useful for reducing
      memory in mixed-precision training.

  Returns:
    A :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def loss(params):
    ...   return jnp.sum(jnp.square(params['w'] - jnp.ones((4, 4))))
    >>> params = {'w': jnp.zeros((4, 4)), 'b': jnp.zeros(4)}
    >>> solver = optax.contrib.mars(learning_rate=3e-3)
    >>> state = solver.init(params)
    >>> for _ in range(5):
    ...   grads = jax.grad(loss)(params)
    ...   updates, state = solver.update(grads, state, params)
    ...   params = optax.apply_updates(params, updates)

  References:
    Yuan et al., `MARS: Unleashing the Power of Variance Reduction for
    Training Large Models <https://arxiv.org/abs/2411.10438>`_, 2024
  """
  return combine.chain(
      scale_by_mars(
          b1=b1,
          b2=b2,
          eps=eps,
          gamma=gamma,
          clip_threshold=clip_threshold,
          mu_dtype=mu_dtype,
      ),
      # pyrefly: ignore[bad-argument-type]
      _adding.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate),
  )
