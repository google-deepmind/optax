# pylint: disable=line-too-long
# pylint: disable=redefined-builtin
"""Utilities for working with tree-like container data structures.

The :mod:`optax.tree` namespace contains aliases of utilities from :mod:`optax.tree_util`.
"""
# pylint: disable=unused-import
from optax.tree_utils._casting import (
tree_cast as cast,
tree_dtype as dtype,
)
from optax.tree_utils._random import (
    tree_random_like as random_like,
    tree_split_key_like as split_key_like,
)

from optax.tree_utils._state_utils import (
    NamedTupleKey,
    tree_get as get,
    tree_get_all_with_path as get_all_with_path,
    tree_map_params as map_params,
    tree_set as set,
)
from optax.tree_utils._tree_math import (
    tree_add as add,
    tree_add_scalar_mul as add_scalar_mul,
    tree_bias_correction as bias_correction,
    tree_clip as clip,
    tree_conj as conj,
    tree_div as div,
    tree_full_like as full_like,
    tree_l1_norm as l1_norm,
    tree_l2_norm as l2_norm,
    tree_linf_norm as linf_norm,
    tree_max as max,
    tree_mul as mul,
    tree_ones_like as ones_like,
    tree_real as real,
    tree_scalar_mul as scalar_mul,
    tree_sub as sub,
    tree_sum as sum,
    tree_update_infinity_moment as update_infinity_moment,
    tree_update_moment as update_moment,
    tree_update_moment_per_elem_norm as update_moment_per_elem_norm,
    tree_vdot as vdot,
    tree_where as where,
    tree_zeros_like as zeros_like
)
