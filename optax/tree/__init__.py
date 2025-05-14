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

# pylint: disable=g-importing-member

from optax.tree_utils import _casting
from optax.tree_utils import _random
from optax.tree_utils import _state_utils
from optax.tree_utils import _tree_math

cast = _casting.tree_cast
cast_like = _casting.tree_cast_like
dtype = _casting.tree_dtype
random_like = _random.tree_random_like
split_key_like = _random.tree_split_key_like
unwrap_random_key_data = _random.tree_unwrap_random_key_data
get = _state_utils.tree_get
get_all_with_path = _state_utils.tree_get_all_with_path
map_params = _state_utils.tree_map_params
set = _state_utils.tree_set  # pylint: disable=redefined-builtin
add = _tree_math.tree_add
add_scale = _tree_math.tree_add_scale
batch_shape = _tree_math.tree_batch_shape
bias_correction = _tree_math.tree_bias_correction
clip = _tree_math.tree_clip
conj = _tree_math.tree_conj
div = _tree_math.tree_div
full_like = _tree_math.tree_full_like
max = _tree_math.tree_max  # pylint: disable=redefined-builtin
mul = _tree_math.tree_mul
norm = _tree_math.tree_norm
ones_like = _tree_math.tree_ones_like
real = _tree_math.tree_real
scale = _tree_math.tree_scale
sub = _tree_math.tree_sub
sum = _tree_math.tree_sum  # pylint: disable=redefined-builtin
update_infinity_moment = _tree_math.tree_update_infinity_moment
update_moment = _tree_math.tree_update_moment
update_moment_per_elem_norm = _tree_math.tree_update_moment_per_elem_norm
vdot = _tree_math.tree_vdot
where = _tree_math.tree_where
zeros_like = _tree_math.tree_zeros_like
