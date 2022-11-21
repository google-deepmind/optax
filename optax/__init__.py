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

from optax import experimental
from optax._src.alias import adabelief
from optax._src.alias import adafactor
from optax._src.alias import adagrad
from optax._src.alias import adam
from optax._src.alias import adamax
from optax._src.alias import adamaxw
from optax._src.alias import adamw
from optax._src.alias import amsgrad
from optax._src.alias import dpsgd
from optax._src.alias import fromage
from optax._src.alias import lamb
from optax._src.alias import lars
from optax._src.alias import MaskOrFn
from optax._src.alias import noisy_sgd
from optax._src.alias import novograd
from optax._src.alias import optimistic_gradient_descent
from optax._src.alias import radam
from optax._src.alias import rmsprop
from optax._src.alias import ScalarOrSchedule
from optax._src.alias import sgd
from optax._src.alias import sm3
from optax._src.alias import yogi
from optax._src.base import EmptyState
from optax._src.base import GradientTransformation
from optax._src.base import identity
from optax._src.base import OptState
from optax._src.base import Params
from optax._src.base import Schedule
from optax._src.base import set_to_zero
from optax._src.base import stateless
from optax._src.base import stateless_with_tree_map
from optax._src.base import TransformInitFn
from optax._src.base import TransformUpdateFn
from optax._src.base import Updates
from optax._src.clipping import adaptive_grad_clip
from optax._src.clipping import AdaptiveGradClipState
from optax._src.clipping import clip
from optax._src.clipping import clip_by_block_rms
from optax._src.clipping import clip_by_global_norm
from optax._src.clipping import ClipByGlobalNormState
from optax._src.clipping import ClipState
from optax._src.clipping import per_example_global_norm_clip
from optax._src.combine import chain
from optax._src.combine import multi_transform
from optax._src.combine import MultiTransformState
from optax._src.constrain import keep_params_nonnegative
from optax._src.constrain import NonNegativeParamsState
from optax._src.constrain import zero_nans
from optax._src.constrain import ZeroNansState
from optax._src.control_variates import control_delta_method
from optax._src.control_variates import control_variates_jacobians
from optax._src.control_variates import moving_avg_baseline
from optax._src.factorized import FactoredState
from optax._src.factorized import scale_by_factored_rms
from optax._src.linear_algebra import global_norm
from optax._src.linear_algebra import matrix_inverse_pth_root
from optax._src.linear_algebra import power_iteration
from optax._src.lookahead import lookahead
from optax._src.lookahead import LookaheadParams
from optax._src.lookahead import LookaheadState
from optax._src.loss import cosine_distance
from optax._src.loss import cosine_similarity
from optax._src.loss import ctc_loss
from optax._src.loss import ctc_loss_with_forward_probs
from optax._src.loss import hinge_loss
from optax._src.loss import huber_loss
from optax._src.loss import l2_loss
from optax._src.loss import log_cosh
from optax._src.loss import sigmoid_binary_cross_entropy
from optax._src.loss import smooth_labels
from optax._src.loss import softmax_cross_entropy
from optax._src.loss import softmax_cross_entropy_with_integer_labels
from optax._src.numerics import safe_int32_increment
from optax._src.numerics import safe_norm
from optax._src.numerics import safe_root_mean_squares
from optax._src.privacy import differentially_private_aggregate
from optax._src.privacy import DifferentiallyPrivateAggregateState
from optax._src.schedule import constant_schedule
from optax._src.schedule import cosine_decay_schedule
from optax._src.schedule import cosine_onecycle_schedule
from optax._src.schedule import exponential_decay
from optax._src.schedule import inject_hyperparams
from optax._src.schedule import InjectHyperparamsState
from optax._src.schedule import join_schedules
from optax._src.schedule import linear_onecycle_schedule
from optax._src.schedule import linear_schedule
from optax._src.schedule import piecewise_constant_schedule
from optax._src.schedule import piecewise_interpolate_schedule
from optax._src.schedule import polynomial_schedule
from optax._src.schedule import sgdr_schedule
from optax._src.schedule import warmup_cosine_decay_schedule
from optax._src.schedule import warmup_exponential_decay_schedule
from optax._src.second_order import fisher_diag
from optax._src.second_order import hessian_diag
from optax._src.second_order import hvp
from optax._src.stochastic_gradient_estimators import measure_valued_jacobians
from optax._src.stochastic_gradient_estimators import pathwise_jacobians
from optax._src.stochastic_gradient_estimators import score_function_jacobians
from optax._src.transform import add_decayed_weights
from optax._src.transform import add_noise
from optax._src.transform import AddDecayedWeightsState
from optax._src.transform import additive_weight_decay
from optax._src.transform import AdditiveWeightDecayState
from optax._src.transform import AddNoiseState
from optax._src.transform import apply_every
from optax._src.transform import ApplyEvery
from optax._src.transform import bias_correction
from optax._src.transform import centralize
from optax._src.transform import ema
from optax._src.transform import EmaState
from optax._src.transform import scale
from optax._src.transform import scale_by_adam
from optax._src.transform import scale_by_adamax
from optax._src.transform import scale_by_amsgrad
from optax._src.transform import scale_by_belief
from optax._src.transform import scale_by_novograd
from optax._src.transform import scale_by_optimistic_gradient
from optax._src.transform import scale_by_param_block_norm
from optax._src.transform import scale_by_param_block_rms
from optax._src.transform import scale_by_radam
from optax._src.transform import scale_by_rms
from optax._src.transform import scale_by_rss
from optax._src.transform import scale_by_schedule
from optax._src.transform import scale_by_sm3
from optax._src.transform import scale_by_stddev
from optax._src.transform import scale_by_trust_ratio
from optax._src.transform import scale_by_yogi
from optax._src.transform import ScaleByAdamState
from optax._src.transform import ScaleByAmsgradState
from optax._src.transform import ScaleByBeliefState
from optax._src.transform import ScaleByFromageState
from optax._src.transform import ScaleByNovogradState
from optax._src.transform import ScaleByRmsState
from optax._src.transform import ScaleByRssState
from optax._src.transform import ScaleByRStdDevState
from optax._src.transform import ScaleByScheduleState
from optax._src.transform import ScaleBySM3State
from optax._src.transform import ScaleByTrustRatioState
from optax._src.transform import ScaleState
from optax._src.transform import trace
from optax._src.transform import TraceState
from optax._src.transform import update_infinity_moment
from optax._src.transform import update_moment
from optax._src.transform import update_moment_per_elem_norm
from optax._src.update import apply_updates
from optax._src.update import incremental_update
from optax._src.update import periodic_update
from optax._src.utils import multi_normal
from optax._src.utils import scale_gradient
from optax._src.wrappers import apply_if_finite
from optax._src.wrappers import ApplyIfFiniteState
from optax._src.wrappers import flatten
from optax._src.wrappers import masked
from optax._src.wrappers import MaskedNode
from optax._src.wrappers import MaskedState
from optax._src.wrappers import maybe_update
from optax._src.wrappers import MaybeUpdateState
from optax._src.wrappers import MultiSteps
from optax._src.wrappers import MultiStepsState
from optax._src.wrappers import ShouldSkipUpdateFunction
from optax._src.wrappers import skip_large_updates
from optax._src.wrappers import skip_not_finite

__version__ = "0.1.4"

__all__ = (
    "adabelief",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamaxw",
    "adamw",
    "adaptive_grad_clip",
    "AdaptiveGradClipState",
    "add_decayed_weights",
    "add_noise",
    "AddDecayedWeightsState",
    "additive_weight_decay",
    "AdditiveWeightDecayState",
    "AddNoiseState",
    "amsgrad",
    "apply_every",
    "apply_if_finite",
    "apply_updates",
    "ApplyEvery",
    "ApplyIfFiniteState",
    "centralize",
    "chain",
    "clip_by_block_rms",
    "clip_by_global_norm",
    "clip",
    "ClipByGlobalNormState",
    "ClipState",
    "constant_schedule",
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "control_delta_method",
    "control_variates_jacobians",
    "cosine_decay_schedule",
    "cosine_distance",
    "cosine_onecycle_schedule",
    "cosine_similarity",
    "differentially_private_aggregate",
    "DifferentiallyPrivateAggregateState",
    "dpsgd",
    "ema",
    "EmaState",
    "EmptyState",
    "exponential_decay",
    "FactoredState",
    "fisher_diag",
    "flatten",
    "fromage",
    "global_norm",
    "GradientTransformation",
    "hinge_loss",
    "hessian_diag",
    "huber_loss",
    "hvp",
    "identity",
    "incremental_update",
    "inject_hyperparams",
    "InjectHyperparamsState",
    "join_schedules",
    "keep_params_nonnegative",
    "l2_loss",
    "lamb",
    "lars",
    "linear_onecycle_schedule",
    "linear_schedule",
    "log_cosh",
    "lookahead",
    "LookaheadParams",
    "LookaheadState",
    "masked",
    "MaskOrFn",
    "MaskedState",
    "matrix_inverse_pth_root",
    "maybe_update",
    "MaybeUpdateState",
    "measure_valued_jacobians",
    "moving_avg_baseline",
    "multi_normal",
    "multi_transform",
    "MultiSteps",
    "MultiStepsState",
    "MultiTransformState",
    "noisy_sgd",
    "novograd",
    "NonNegativeParamsState",
    "OptState",
    "Params",
    "pathwise_jacobians",
    "periodic_update",
    "per_example_global_norm_clip",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polynomial_schedule",
    "power_iteration",
    "radam",
    "rmsprop",
    "safe_int32_increment",
    "safe_norm",
    "safe_root_mean_squares",
    "ScalarOrSchedule",
    "scale_by_adam",
    "scale_by_adamax",
    "scale_by_amsgrad",
    "scale_by_belief",
    "scale_by_factored_rms",
    "scale_by_novograd",
    "scale_by_param_block_norm",
    "scale_by_param_block_rms",
    "scale_by_radam",
    "scale_by_rms",
    "scale_by_rss",
    "scale_by_schedule",
    "scale_by_sm3",
    "scale_by_stddev",
    "scale_by_trust_ratio",
    "scale_by_yogi",
    "scale_gradient",
    "scale",
    "ScaleByAdamState",
    "ScaleByAmsgradState",
    "ScaleByBeliefState",
    "ScaleByFromageState",
    "ScaleByNovogradState",
    "ScaleByRmsState",
    "ScaleByRssState",
    "ScaleByRStdDevState",
    "ScaleByScheduleState",
    "ScaleBySM3State",
    "ScaleByTrustRatioState",
    "ScaleState",
    "Schedule",
    "score_function_jacobians",
    "set_to_zero",
    "sgd",
    "sgdr_schedule",
    "ShouldSkipUpdateFunction",
    "sigmoid_binary_cross_entropy",
    "skip_large_updates",
    "skip_not_finite",
    "sm3",
    "smooth_labels",
    "softmax_cross_entropy",
    "stateless",
    "stateless_with_tree_map",
    "trace",
    "TraceState",
    "TransformInitFn",
    "TransformUpdateFn",
    "Updates",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
    "yogi",
    "zero_nans",
    "ZeroNansState",
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
