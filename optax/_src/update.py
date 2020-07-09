# Lint as: python3
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
"""Apply transformed gradient updates to parameters."""

import jax
from optax._src import transform


def apply_updates(
    params: transform.Params, updates: transform.Updates) -> transform.Params:
  """Applies an update to the corresponding parameters.

  This is an (optional) utility functions that applies an update, and returns
  the updated parameters to the caller. The update itself is typically the
  result of applying any number of `chainable` transformations.

  Args:
    params: a tree of parameters.
    updates: a tree of updates, the tree structure and the shape of the leaf
    nodes must match that of `params`.

  Returns:
    Updated parameters, with same structure and shape as `params`.
  """
  return jax.tree_multimap(lambda p, u: p + u, params, updates)

