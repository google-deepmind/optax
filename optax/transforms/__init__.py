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

# pylint: disable=g-importing-member

from optax.transforms._accumulation import ema as ema
from optax.transforms._accumulation import EmaState as EmaState
from optax.transforms._accumulation import MultiSteps as MultiSteps
from optax.transforms._accumulation import MultiStepsState as MultiStepsState
from optax.transforms._accumulation import ShouldSkipUpdateFunction as ShouldSkipUpdateFunction
from optax.transforms._accumulation import skip_large_updates as skip_large_updates
from optax.transforms._accumulation import skip_not_finite as skip_not_finite
from optax.transforms._accumulation import trace as trace
from optax.transforms._accumulation import TraceState as TraceState
from optax.transforms._adding import add_decayed_weights as add_decayed_weights
from optax.transforms._adding import add_noise as add_noise
from optax.transforms._adding import AddNoiseState as AddNoiseState
from optax.transforms._clipping import adaptive_grad_clip as adaptive_grad_clip
from optax.transforms._clipping import clip as clip
from optax.transforms._clipping import clip_by_block_rms as clip_by_block_rms
from optax.transforms._clipping import clip_by_global_norm as clip_by_global_norm
from optax.transforms._clipping import per_example_global_norm_clip as per_example_global_norm_clip
from optax.transforms._clipping import per_example_layer_norm_clip as per_example_layer_norm_clip
from optax.transforms._clipping import unitwise_clip as unitwise_clip
from optax.transforms._clipping import unitwise_norm as unitwise_norm
from optax.transforms._combining import chain as chain
from optax.transforms._combining import named_chain as named_chain
from optax.transforms._combining import partition as partition
from optax.transforms._combining import PartitionState as PartitionState
from optax.transforms._conditionality import apply_if_finite as apply_if_finite
from optax.transforms._conditionality import ApplyIfFiniteState as ApplyIfFiniteState
from optax.transforms._conditionality import conditionally_mask as conditionally_mask
from optax.transforms._conditionality import conditionally_transform as conditionally_transform
from optax.transforms._conditionality import ConditionallyMaskState as ConditionallyMaskState
from optax.transforms._conditionality import ConditionallyTransformState as ConditionallyTransformState
from optax.transforms._conditionality import ConditionFn as ConditionFn
from optax.transforms._constraining import keep_params_nonnegative as keep_params_nonnegative
from optax.transforms._constraining import NonNegativeParamsState as NonNegativeParamsState
from optax.transforms._constraining import zero_nans as zero_nans
from optax.transforms._constraining import ZeroNansState as ZeroNansState
from optax.transforms._layouts import flatten as flatten
from optax.transforms._masking import masked as masked
from optax.transforms._masking import MaskedNode as MaskedNode
from optax.transforms._masking import MaskedState as MaskedState
