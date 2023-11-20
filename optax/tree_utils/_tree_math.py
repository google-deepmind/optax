# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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

"""Utilities to perform maths on pytrees."""

import functools
import operator
from typing import Any

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x: Any, tree_y: Any) -> chex.Numeric:
  r"""Compute the inner product between two pytrees.

  Args:
    tree_x: first pytree to use.
    tree_y: second pytree to use.
  Returns:
    inner product between ``tree_x`` and ``tree_y``, a scalar value.

  >>> optax.tree_utils.tree_vdot(
  >>>   {a: jnp.array([1, 2]), b: jnp.array([1, 2])},
  >>>   {a: jnp.array([-1, -1]), b: jnp.array([1, 1])},
  >>> )
  0.0

  Implementation detail: we upcast the values to the highest precision to avoid
  numerical issues.
  """
  vdots = jtu.tree_map(_vdot_safe, tree_x, tree_y)
  return jtu.tree_reduce(operator.add, vdots)
