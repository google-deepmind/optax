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
"""Cautious optimizer wrapper.

Reference:
  Liang et al., `Cautious Optimizers: Improving Training with One Line of Code
  <https://arxiv.org/abs/2411.16085>`_, 2024.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from optax._src import base


class CautiousState(NamedTuple):
  """State for the :func:`optax.contrib.cautious` wrapper."""

  base_optimizer_state: base.OptState


def cautious(
    base_optimizer: base.GradientTransformation,
    eps: float = 1e-8,
) -> base.GradientTransformationExtraArgs:
  r"""Cautious wrapper: mask updates that disagree with the current gradient.

  Wraps an arbitrary ``base_optimizer`` and, on every step, zeroes the
  coordinates of the proposed update that would move *against* the current
  gradient (i.e. would locally *increase* the loss), then rescales the
  surviving coordinates so the average update magnitude is preserved.

  Concretely, let :math:`u_t` be the update proposed by the base optimizer
  (using Optax's additive convention ``params <- params + u_t``) and
  :math:`g_t` the current gradient. The cautious mask keeps only the
  descent-aligned coordinates:

  .. math::

    \phi_t = \mathbb{1}\!\left[u_t \odot g_t < 0\right],

  and rescales them per parameter tensor so the mean magnitude is unchanged:

  .. math::

    \tilde{u}_t = \phi_t \odot u_t \cdot \frac{n}{\sum \phi_t + \varepsilon},

  where :math:`n` is the number of elements of the tensor. The mask
  condition :math:`u_t \odot g_t < 0` is exactly the paper's alignment
  condition :math:`(-u_t) \odot g_t > 0` re-expressed in Optax's additive
  update convention (Optax updates are the negative of the paper's, since
  Optax *adds* the update while the paper *subtracts* it).

  This single-line modification provably preserves the Hamiltonian / Lyapunov
  descent of the base optimizer: the cautious update always satisfies
  :math:`\langle \tilde{u}_t, g_t \rangle \le 0`, so it never points uphill,
  whereas a momentum-based base optimizer can. Empirically the authors report
  up to a 1.47x sample-efficiency gain when wrapping AdamW for LLM and ViT
  pre-training, at the cost of one elementwise mask.

  Because the mask needs *both* the raw gradient and the base optimizer's
  proposed update, ``cautious`` is implemented as a wrapper (like
  :func:`optax.contrib.schedule_free`) rather than a chainable
  ``scale_by_*`` transform.

  Args:
    base_optimizer: The optimizer to wrap (e.g. ``optax.adamw(1e-3)``,
      ``optax.lion(1e-4)``, or any :class:`optax.GradientTransformation`).
    eps: Small constant in the rescaling denominator. With the default
      ``1e-8`` the wrapper reduces *exactly* to ``base_optimizer`` when every
      coordinate agrees with the gradient (the mean-preserving normalization).
      The original paper uses ``eps=1``, which additionally damps the update
      when only a few coordinates survive; pass ``eps=1.0`` to match it.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic objective
    >>> base = optax.adamw(learning_rate=0.1)
    >>> solver = optax.contrib.cautious(base)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...   grad = jax.grad(f)(params)
    ...   updates, opt_state = solver.update(grad, opt_state, params)
    ...   params = optax.apply_updates(params, updates)
    ...   print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.28E+01
    Objective function: 1.17E+01
    Objective function: 1.07E+01
    Objective function: 9.69E+00
    Objective function: 8.77E+00

  References:
    Liang et al, `Cautious Optimizers: Improving Training with One Line of Code
    <https://arxiv.org/abs/2411.16085>`_, 2024.
  """
  base_optimizer = base.with_extra_args_support(base_optimizer)

  def init_fn(params: base.Params) -> CautiousState:
    return CautiousState(base_optimizer_state=base_optimizer.init(params))

  def update_fn(
      updates: base.Updates,
      state: CautiousState,
      params: Optional[base.Params] = None,
      **extra_args,
  ):
    # ``updates`` are the raw gradients fed to the wrapper.
    grads = updates
    base_updates, new_base_state = base_optimizer.update(
        grads, state.base_optimizer_state, params, **extra_args
    )

    def _mask_leaf(update_leaf, grad_leaf):
      # Keep coordinates where the update opposes the gradient (descent in the
      # additive Optax convention ``params <- params + update``).
      keep = (update_leaf * grad_leaf < 0).astype(update_leaf.dtype)
      # Per-tensor mean-preserving rescale.
      scale = keep.size / (jnp.sum(keep) + eps)
      return update_leaf * keep * scale

    cautious_updates = jax.tree.map(_mask_leaf, base_updates, grads)
    return cautious_updates, CautiousState(base_optimizer_state=new_base_state)

  # pyrefly: ignore[bad-argument-type]
  return base.GradientTransformationExtraArgs(init_fn, update_fn)
