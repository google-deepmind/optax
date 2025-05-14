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
"""The tree_utils sub-package."""

import functools
import typing

# pylint: disable=g-importing-member

from optax.tree_utils._casting import tree_cast
from optax.tree_utils._casting import tree_cast_like
from optax.tree_utils._casting import tree_dtype
from optax.tree_utils._random import tree_random_like
from optax.tree_utils._random import tree_split_key_like
from optax.tree_utils._random import tree_unwrap_random_key_data
from optax.tree_utils._state_utils import NamedTupleKey
from optax.tree_utils._state_utils import tree_get
from optax.tree_utils._state_utils import tree_get_all_with_path
from optax.tree_utils._state_utils import tree_map_params
from optax.tree_utils._state_utils import tree_set
from optax.tree_utils._tree_math import tree_add
from optax.tree_utils._tree_math import tree_add_scale
from optax.tree_utils._tree_math import tree_batch_shape
from optax.tree_utils._tree_math import tree_bias_correction
from optax.tree_utils._tree_math import tree_clip
from optax.tree_utils._tree_math import tree_conj
from optax.tree_utils._tree_math import tree_div
from optax.tree_utils._tree_math import tree_full_like
from optax.tree_utils._tree_math import tree_max
from optax.tree_utils._tree_math import tree_mul
from optax.tree_utils._tree_math import tree_norm
from optax.tree_utils._tree_math import tree_ones_like
from optax.tree_utils._tree_math import tree_real
from optax.tree_utils._tree_math import tree_scale
from optax.tree_utils._tree_math import tree_sub
from optax.tree_utils._tree_math import tree_sum
from optax.tree_utils._tree_math import tree_update_infinity_moment
from optax.tree_utils._tree_math import tree_update_moment
from optax.tree_utils._tree_math import tree_update_moment_per_elem_norm
from optax.tree_utils._tree_math import tree_vdot
from optax.tree_utils._tree_math import tree_where
from optax.tree_utils._tree_math import tree_zeros_like

_deprecations = {
    # Added Mar 2025
    'tree_scalar_mul': (
        ('optax.tree_utils.tree_scalar_mul is deprecated: use'
         ' optax.tree_utils.tree_scale (optax v0.2.5 or newer).'),
        tree_scale,
    ),
    'tree_add_scalar_mul': (
        ('optax.tree_utils.tree_scalar_mul is deprecated: use'
         ' optax.tree_utils.tree_scale (optax v0.2.5 or newer).'),
        tree_add_scale,
    ),
    # Added May 2025
    'tree_l1_norm': (
        ('optax.tree_utils.tree_l1_norm is deprecated: use'
         ' optax.tree_utils.tree_norm(..., ord=1) (optax v0.2.5 or newer).'),
        functools.partial(tree_norm, ord=1),
    ),
    'tree_l2_norm': (
        ('optax.tree_utils.tree_l2_norm is deprecated: use'
         ' optax.tree_utils.tree_norm (optax v0.2.5 or newer).'),
        functools.partial(tree_norm, ord=2),
    ),
    'tree_linf_norm': (
        ('optax.tree_utils.tree_linf_norm is deprecated: use'
         ' optax.tree_utils.tree_norm(..., ord=jnp.inf)'
         ' (optax v0.2.5 or newer).'),
        functools.partial(tree_norm, ord='inf'),
    ),
}

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
if typing.TYPE_CHECKING:
  tree_scalar_mul = tree_scale
  tree_add_scalar_mul = tree_add_scale
  tree_l1_norm = functools.partial(tree_norm, ord=1)
  tree_l2_norm = tree_norm
  tree_linf_norm = functools.partial(tree_norm, ord='inf')

else:
  from optax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
