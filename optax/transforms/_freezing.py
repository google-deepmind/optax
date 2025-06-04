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

"""Utilites for freezing parameters."""

from typing import Union

import chex
import jax

from optax._src import base
# pylint: disable=g-importing-member
from optax.transforms._combining import partition
from optax.transforms._masking import masked
# pylint: enable=g-importing-member


def freeze(mask: Union[bool, chex.ArrayTree]) -> base.GradientTransformation:
  """Create a transformation that zeros out gradient updates for `mask=True`.

  This essentially freezes (i.e. holding constant) masked parameters.

  The mask must be static (i.e., not dependent on runtime values or updated
  during training) and can be:
    * a single boolean (or 0-d JAX bool array), causing every parameter to be
      either all-frozen (True) or all-trainable (False), or
    * a PyTree of booleans matching the structure of the parameters, where
      each leaf indicates whether that specific parameter leaf should be
      frozen (True) or left unchanged (False).

  Args:
    mask: A boolean prefix tree mask indicating which parameters to freeze.

  Example:
    >>> params = {'a': jnp.zeros(1), 'b': jnp.zeros(2)}
    >>> mask = {'a': True, 'b': False} # Freeze 'a', train 'b'
    >>> freezer = freeze(mask)

  Returns:
    An Optax `GradientTransformation` which applies `set_to_zero()` wherever
    `mask==True`, and leaves other gradients intact.

  .. seealso::
      :func:`optax.transforms.selective_transform` : For partitioning updates
      so only un-frozen parameters are optimized.
  """
  return masked(base.set_to_zero(), mask)


def selective_transform(
    optimizer: base.GradientTransformation,
    *,  # force kw-only arguments to show this is a freeze and not allow mask
    freeze_mask: Union[bool, chex.ArrayTree],
) -> base.GradientTransformation:
  """Partition updates so that only un-frozen parameters are optimized.

  Example:
    >>> params = {'a': jnp.zeros(1), 'b': jnp.zeros(2)}
    >>> mask = {'a': True, 'b': False} # Freeze 'a', train 'b'
    >>> selective_opt = selective_transform(optax.adam(1e-3), mask)

  Args:
    optimizer: The inner Optax optimizer to apply to unfrozen leaves.
    freeze_mask: A *static* mask (i.e., not dependent on runtime values or
      updated during training). It can be either:
        * a scalar bool (or 0-d JAX bool array) to freeze everything (True) or
          nothing (False)
        * a PyTree of booleans mirroring the parameter tree, marking each leaf
          to freeze (True) or train (False).

  Returns:
    A `GradientTransformation` that routes each parameter leaf through:
      * the given `optimizer` if its mask is False (“train”),
      * `set_to_zero()` if its mask is True (“freeze”).

  .. seealso::
      :func:`optax.transforms.freeze` : For simply zeroing out gradients
      according to a mask.
  """

  def label_fn(params: base.PyTree):
    del params
    return jax.tree.map(lambda m: "freeze" if m else "train", freeze_mask)

  return partition(
      {"train": optimizer, "freeze": base.set_to_zero()},
      param_labels=label_fn,
  )
