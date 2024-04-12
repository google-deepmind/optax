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
"""The transforms sub-package."""

from optax.transforms._accumulation import ema
from optax.transforms._accumulation import EmaState
from optax.transforms._accumulation import MultiSteps
from optax.transforms._accumulation import MultiStepsState
from optax.transforms._accumulation import ShouldSkipUpdateFunction
from optax.transforms._accumulation import skip_large_updates
from optax.transforms._accumulation import skip_not_finite
from optax.transforms._accumulation import trace
from optax.transforms._accumulation import TraceState
from optax.transforms._conditionality import apply_if_finite
from optax.transforms._conditionality import ApplyIfFiniteState
from optax.transforms._conditionality import conditionally_mask
from optax.transforms._conditionality import conditionally_transform
from optax.transforms._conditionality import ConditionallyMaskState
from optax.transforms._conditionality import ConditionallyTransformState
from optax.transforms._conditionality import ConditionFn
from optax.transforms._layouts import flatten
from optax.transforms._masking import masked
from optax.transforms._masking import MaskedNode
from optax.transforms._masking import MaskedState
