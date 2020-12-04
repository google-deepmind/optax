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
"""Optax: composable gradient processing and optimization, in JAX."""

from optax._src.alias import adabelief
from optax._src.alias import adagrad
from optax._src.alias import adam
from optax._src.alias import adamw
from optax._src.alias import lamb
from optax._src.alias import noisy_sgd
from optax._src.alias import rmsprop
from optax._src.alias import sgd
from optax._src.combine import chain
from optax._src.control_variates import control_delta_method
from optax._src.control_variates import control_variates_jacobians
from optax._src.control_variates import moving_avg_baseline
from optax._src.schedule import constant_schedule
from optax._src.schedule import cosine_decay_schedule
from optax._src.schedule import exponential_decay
from optax._src.schedule import piecewise_constant_schedule
from optax._src.schedule import polynomial_schedule
from optax._src.second_order import fisher_diag
from optax._src.second_order import hessian_diag
from optax._src.second_order import hvp
from optax._src.stochastic_gradient_estimators import measure_valued_jacobians
from optax._src.stochastic_gradient_estimators import pathwise_jacobians
from optax._src.stochastic_gradient_estimators import score_function_jacobians
from optax._src.transform import add_noise
from optax._src.transform import additive_weight_decay
from optax._src.transform import AdditiveWeightDecayState
from optax._src.transform import AddNoiseState
from optax._src.transform import apply_every
from optax._src.transform import ApplyEvery
from optax._src.transform import centralize
from optax._src.transform import clip
from optax._src.transform import clip_by_global_norm
from optax._src.transform import ClipByGlobalNormState
from optax._src.transform import ClipState
from optax._src.transform import global_norm
from optax._src.transform import GradientTransformation
from optax._src.transform import identity
from optax._src.transform import InitUpdate  # To be removed
from optax._src.transform import OptState
from optax._src.transform import Params
from optax._src.transform import scale
from optax._src.transform import scale_by_adam
from optax._src.transform import scale_by_belief
from optax._src.transform import scale_by_fromage
from optax._src.transform import scale_by_rms
from optax._src.transform import scale_by_rss
from optax._src.transform import scale_by_schedule
from optax._src.transform import scale_by_stddev
from optax._src.transform import scale_by_trust_ratio
from optax._src.transform import ScaleByAdamState
from optax._src.transform import ScaleByFromageState
from optax._src.transform import ScaleByRmsState
from optax._src.transform import ScaleByRssState
from optax._src.transform import ScaleByRStdDevState
from optax._src.transform import ScaleByScheduleState
from optax._src.transform import ScaleByTrustRatioState
from optax._src.transform import ScaleState
from optax._src.transform import trace
from optax._src.transform import TraceState
from optax._src.transform import Updates
from optax._src.update import apply_updates
from optax._src.utils import multi_normal
from optax._src.wrappers import apply_if_finite
from optax._src.wrappers import ApplyIfFiniteState
from optax._src.wrappers import flatten

__version__ = "0.0.2"

__all__ = (
    "adabelief",
    "adagrad",
    "adam",
    "adamw",
    "add_noise",
    "additive_weight_decay",
    "AdditiveWeightDecayState",
    "AddNoiseState",
    "apply_if_finite",
    "apply_every",
    "apply_updates",
    "ApplyEvery",
    "ApplyIfFiniteState",
    "centralize",
    "chain",
    "clip",
    "clip_by_global_norm",
    "ClipByGlobalNormState",
    "ClipState",
    "constant_schedule",
    "control_delta_method",
    "control_variates_jacobians",
    "cosine_decay_schedule",
    "exponential_decay",
    "fisher_diag",
    "flatten",
    "global_norm",
    "GradientTransformation",
    "hessian_diag",
    "hvp",
    "identity",
    "InitUpdate",
    "lamb",
    "measure_valued_jacobians",
    "moving_avg_baseline",
    "multi_normal",
    "noisy_sgd",
    "OptState",
    "Params",
    "pathwise_jacobians",
    "piecewise_constant_schedule",
    "polynomial_schedule",
    "rmsprop",
    "scale",
    "scale_by_adam",
    "scale_by_belief",
    "scale_by_fromage",
    "scale_by_rms",
    "scale_by_rss",
    "scale_by_schedule",
    "scale_by_stddev",
    "scale_by_trust_ratio",
    "ScaleByAdamState",
    "ScaleByFromageState",
    "ScaleByRmsState",
    "ScaleByRssState",
    "ScaleByRStdDevState",
    "ScaleByScheduleState",
    "ScaleByTrustRatioState",
    "ScaleState",
    "score_function_jacobians",
    "sgd",
    "trace",
    "TraceState",
    "Updates",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Optax public API.   /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
