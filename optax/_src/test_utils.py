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
"""Testing utilities for Optax."""

import inspect
import types
from typing import Sequence, Tuple


def find_internal_python_modules(
    root_module: types.ModuleType,
) -> Sequence[Tuple[str, types.ModuleType]]:
  """Returns `(name, module)` for all Optax submodules under `root_module`."""
  modules = set([(root_module.__name__, root_module)])
  visited = set()
  to_visit = [root_module]

  while to_visit:
    mod = to_visit.pop()
    visited.add(mod)

    for name in dir(mod):
      obj = getattr(mod, name)
      if inspect.ismodule(obj) and obj not in visited:
        if obj.__name__.startswith('optax'):
          if '_src' not in obj.__name__:
            to_visit.append(obj)
            modules.add((obj.__name__, obj))

  return sorted(modules)
