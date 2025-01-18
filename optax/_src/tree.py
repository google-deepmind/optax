"""Utilities for working with tree-like container data structures. The 
:mod:`optax.tree` namespace contains aliases of utilities from 
:mod:`optax.tree_util`."""

# pylint: disable=unused-import
from optax.tree_utils._casting import tree_cast as cast
from optax.tree_utils._casting import tree_dtype as dtype

from optax.tree_utils._random import tree_random_like as random_like
from optax.tree_utils._random import tree_split_key_like as split_key_like

from optax.tree_utils._state_utils import NamedTupleKey
from optax.tree_utils._state_utils import tree_get as get
from optax.tree_utils._state_utils import tree_get_all_with_path as get_all_with_path
from optax.tree_utils._state_utils import tree_map_params as map_params
from optax.tree_utils._state_utils import tree_set as set # pylint: disable=redefined-builtin

from optax.tree_utils._tree_math import tree_add as add
from optax.tree_utils._tree_math import tree_add_scalar_mul as add_scalar_mul
from optax.tree_utils._tree_math import tree_bias_correction as bias_correction
from optax.tree_utils._tree_math import tree_clip as clip
from optax.tree_utils._tree_math import tree_conj as conj
from optax.tree_utils._tree_math import tree_div as div
from optax.tree_utils._tree_math import tree_full_like as full_like
from optax.tree_utils._tree_math import tree_l1_norm as l1_norm
from optax.tree_utils._tree_math import tree_l2_norm as l2_norm
from optax.tree_utils._tree_math import tree_linf_norm as linf_norm
from optax.tree_utils._tree_math import tree_max as max # pylint: disable=redefined-builtin
from optax.tree_utils._tree_math import tree_mul as mul
from optax.tree_utils._tree_math import tree_ones_like as ones_like
from optax.tree_utils._tree_math import tree_real as real
from optax.tree_utils._tree_math import tree_scalar_mul as scalar_mul
from optax.tree_utils._tree_math import tree_sub as sub
from optax.tree_utils._tree_math import tree_sum as sum # pylint: disable=redefined-builtin
from optax.tree_utils._tree_math import tree_update_infinity_moment as update_infinity_moment
from optax.tree_utils._tree_math import tree_update_moment as update_moment
from optax.tree_utils._tree_math import tree_update_moment_per_elem_norm as update_moment_per_elem_norm
from optax.tree_utils._tree_math import tree_vdot as vdot
from optax.tree_utils._tree_math import tree_where as where
from optax.tree_utils._tree_math import tree_zeros_like as zeros_like

