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
"""Complex-valued optimization."""

from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from optax._src import base


@jax.tree_util.register_pytree_node_class
@dataclass
class RealPair:
  """A pair of real arrays split from a complex array."""
  real: chex.Array
  imag: chex.Array

  def tree_flatten(self):
    return ((self.real, self.imag), None)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


def _is_real_pair(x):
  return isinstance(x, RealPair)


def _complex_to_real_pair(x):
  if jnp.iscomplexobj(x):
    return RealPair(x.real, x.imag)
  else:
    return x


def _real_pair_to_complex(x):
  if _is_real_pair(x):
    return x.real + x.imag * 1j
  else:
    return x


class SplitComplexState(NamedTuple):
  """Maintains the inner transformation state for `split_complex`."""
  inner_state: base.OptState


def split_complex(
    inner: base.GradientTransformation
) -> base.GradientTransformation:
  """Splits complex parameters into pairs of real parameters.

  The inner transformation processes real parameters and updates, and the
  pairs of transformed real updates are merged into complex updates.

  Parameters that are real before `split_complex` are passed through unmodified.

  Args:
    inner: The inner transformation.

  Returns:
    An `optax.GradientTransformation`.
  """

  def init_fn(params):
    params = jax.tree_map(_complex_to_real_pair, params)
    inner_state = inner.init(params)
    return SplitComplexState(inner_state)

  def update_fn(updates, state, params=None):
    inner_state = state.inner_state
    updates = jax.tree_map(_complex_to_real_pair, updates)
    params = jax.tree_map(_complex_to_real_pair, params)
    updates, inner_state = inner.update(updates, inner_state, params)
    updates = jax.tree_map(
        _real_pair_to_complex, updates, is_leaf=_is_real_pair)
    return updates, SplitComplexState(inner_state)

  return base.GradientTransformation(init_fn, update_fn)
