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
"""Import stub."""

# TODO(mtthss): delete this file asap.
import warnings
from optax.schedules import _inject

# warn that this is a deprecated file
warnings.warn(
    "module optax.schedules.inject is deprecated. Please use optax.schedules"
    " instead",
    DeprecationWarning,
    stacklevel=2,
)

InjectHyperparamsState = _inject.InjectHyperparamsState
inject_hyperparams = _inject.inject_hyperparams
