# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Euclidean projections."""

from typing import Any

import jax
from jax import tree_util as jtu
import jax.numpy as jnp


def projection_non_negative(pytree: Any) -> Any:
  r"""Projection onto the non-negative orthant.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  where :math:`x` is the input pytree.

  Args:
    pytree: pytree to project.
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return jtu.tree_map(jax.nn.relu, pytree)


def _clip_safe(leaf, lower, upper):
  return jnp.clip(jnp.asarray(leaf), lower, upper)


def projection_box(pytree: Any, lower: Any, upper: Any) -> Any:
  r"""Projection onto box constraints.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    \text{lower} \le p \le \text{upper}

  where :math:`x` is the input pytree.

  Args:
    pytree: pytree to project.
    lower:  lower bound, a scalar or pytree with the same structure as
      ``pytree``.
    upper:  upper bound, a scalar or pytree with the same structure as
      ``pytree``.
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return jtu.tree_map(_clip_safe, pytree, lower, upper)


def projection_hypercube(pytree: Any, scale: Any = 1.0) -> Any:
  r"""Projection onto the (unit) hypercube.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    0 \le p \le \text{scale}

  where :math:`x` is the input pytree.

  By default, we project to the unit hypercube (`scale=1.0`).

  This is a convenience wrapper around
  :func:`projection_box <optax.projections.projection_box>`.

  Args:
    pytree: pytree to project.
    scale: scale of the hypercube, a scalar or a pytree (default: 1.0).
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return projection_box(pytree, lower=0.0, upper=scale)
