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
"""Utilities to cast pytrees to specific dtypes."""

from typing import Optional

import chex
from jax import tree_util as jtu


def tree_cast(
    tree: chex.ArrayTree,
    dtype: Optional[chex.ArrayDType]
) -> chex.ArrayTree:
  """Cast tree to given dtype, skip if None."""
  if dtype is not None:
    return jtu.tree_map(lambda t: t.astype(dtype), tree)
  else:
    return tree
