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

# pylint: disable=wrong-import-position
# pylint: disable=g-importing-member

import typing as _typing

from optax import assignment as assignment
from optax import contrib as contrib
from optax import losses as losses
from optax import monte_carlo as monte_carlo
from optax import perturbations as perturbations
from optax import projections as projections
from optax import schedules as schedules
from optax import second_order as second_order
from optax import transforms as transforms
from optax import tree_utils as tree_utils
from optax._src.alias import adabelief as adabelief
from optax._src.alias import adadelta as adadelta
from optax._src.alias import adafactor as adafactor
from optax._src.alias import adagrad as adagrad
from optax._src.alias import adam as adam
from optax._src.alias import adamax as adamax
from optax._src.alias import adamaxw as adamaxw
from optax._src.alias import adamw as adamw
from optax._src.alias import adan as adan
from optax._src.alias import amsgrad as amsgrad
from optax._src.alias import fromage as fromage
from optax._src.alias import lamb as lamb
from optax._src.alias import lars as lars
from optax._src.alias import lbfgs as lbfgs
from optax._src.alias import lion as lion
from optax._src.alias import MaskOrFn as MaskOrFn
from optax._src.alias import nadam as nadam
from optax._src.alias import nadamw as nadamw
from optax._src.alias import noisy_sgd as noisy_sgd
from optax._src.alias import novograd as novograd
from optax._src.alias import optimistic_adam as optimistic_adam
from optax._src.alias import optimistic_gradient_descent as optimistic_gradient_descent
from optax._src.alias import polyak_sgd as polyak_sgd
from optax._src.alias import radam as radam
from optax._src.alias import rmsprop as rmsprop
from optax._src.alias import rprop as rprop
from optax._src.alias import sgd as sgd
from optax._src.alias import sign_sgd as sign_sgd
from optax._src.alias import sm3 as sm3
from optax._src.alias import yogi as yogi
from optax._src.base import EmptyState as EmptyState
from optax._src.base import GradientTransformation as GradientTransformation
from optax._src.base import GradientTransformationExtraArgs as GradientTransformationExtraArgs
from optax._src.base import identity as identity
from optax._src.base import OptState as OptState
from optax._src.base import Params as Params
from optax._src.base import ScalarOrSchedule as ScalarOrSchedule
from optax._src.base import Schedule as Schedule
from optax._src.base import set_to_zero as set_to_zero
from optax._src.base import stateless as stateless
from optax._src.base import stateless_with_tree_map as stateless_with_tree_map
from optax._src.base import TransformInitFn as TransformInitFn
from optax._src.base import TransformUpdateExtraArgsFn as TransformUpdateExtraArgsFn
from optax._src.base import TransformUpdateFn as TransformUpdateFn
from optax._src.base import Updates as Updates
from optax._src.base import with_extra_args_support as with_extra_args_support
from optax._src.factorized import FactoredState as FactoredState
from optax._src.factorized import scale_by_factored_rms as scale_by_factored_rms
from optax._src.linear_algebra import global_norm as global_norm
from optax._src.linear_algebra import matrix_inverse_pth_root as matrix_inverse_pth_root
from optax._src.linear_algebra import power_iteration as power_iteration
from optax._src.linesearch import scale_by_backtracking_linesearch as scale_by_backtracking_linesearch
from optax._src.linesearch import scale_by_zoom_linesearch as scale_by_zoom_linesearch
from optax._src.linesearch import ScaleByBacktrackingLinesearchState as ScaleByBacktrackingLinesearchState
from optax._src.linesearch import ScaleByZoomLinesearchState as ScaleByZoomLinesearchState
from optax._src.linesearch import ZoomLinesearchInfo as ZoomLinesearchInfo
from optax._src.lookahead import lookahead as lookahead
from optax._src.lookahead import LookaheadParams as LookaheadParams
from optax._src.lookahead import LookaheadState as LookaheadState
from optax._src.numerics import safe_increment as safe_increment
from optax._src.numerics import safe_int32_increment as safe_int32_increment
from optax._src.numerics import safe_norm as safe_norm
from optax._src.numerics import safe_root_mean_squares as safe_root_mean_squares
from optax._src.transform import apply_every as apply_every
from optax._src.transform import ApplyEvery as ApplyEvery
from optax._src.transform import centralize as centralize
from optax._src.transform import normalize_by_update_norm as normalize_by_update_norm
from optax._src.transform import scale as scale
from optax._src.transform import scale_by_adadelta as scale_by_adadelta
from optax._src.transform import scale_by_adam as scale_by_adam
from optax._src.transform import scale_by_adamax as scale_by_adamax
from optax._src.transform import scale_by_adan as scale_by_adan
from optax._src.transform import scale_by_amsgrad as scale_by_amsgrad
from optax._src.transform import scale_by_belief as scale_by_belief
from optax._src.transform import scale_by_distance_over_gradients as scale_by_distance_over_gradients
from optax._src.transform import scale_by_lbfgs as scale_by_lbfgs
from optax._src.transform import scale_by_learning_rate as scale_by_learning_rate
from optax._src.transform import scale_by_lion as scale_by_lion
from optax._src.transform import scale_by_novograd as scale_by_novograd
from optax._src.transform import scale_by_optimistic_gradient as scale_by_optimistic_gradient
from optax._src.transform import scale_by_param_block_norm as scale_by_param_block_norm
from optax._src.transform import scale_by_param_block_rms as scale_by_param_block_rms
from optax._src.transform import scale_by_polyak as scale_by_polyak
from optax._src.transform import scale_by_radam as scale_by_radam
from optax._src.transform import scale_by_rms as scale_by_rms
from optax._src.transform import scale_by_rprop as scale_by_rprop
from optax._src.transform import scale_by_rss as scale_by_rss
from optax._src.transform import scale_by_schedule as scale_by_schedule
from optax._src.transform import scale_by_sign as scale_by_sign
from optax._src.transform import scale_by_sm3 as scale_by_sm3
from optax._src.transform import scale_by_stddev as scale_by_stddev
from optax._src.transform import scale_by_trust_ratio as scale_by_trust_ratio
from optax._src.transform import scale_by_yogi as scale_by_yogi
from optax._src.transform import ScaleByAdaDeltaState as ScaleByAdaDeltaState
from optax._src.transform import ScaleByAdamState as ScaleByAdamState
from optax._src.transform import ScaleByAdanState as ScaleByAdanState
from optax._src.transform import ScaleByAmsgradState as ScaleByAmsgradState
from optax._src.transform import ScaleByBeliefState as ScaleByBeliefState
from optax._src.transform import ScaleByLBFGSState as ScaleByLBFGSState
from optax._src.transform import ScaleByLionState as ScaleByLionState
from optax._src.transform import ScaleByNovogradState as ScaleByNovogradState
from optax._src.transform import ScaleByRmsState as ScaleByRmsState
from optax._src.transform import ScaleByRpropState as ScaleByRpropState
from optax._src.transform import ScaleByRssState as ScaleByRssState
from optax._src.transform import ScaleByRStdDevState as ScaleByRStdDevState
from optax._src.transform import ScaleByScheduleState as ScaleByScheduleState
from optax._src.transform import ScaleBySM3State as ScaleBySM3State
from optax._src.update import apply_updates as apply_updates
from optax._src.update import incremental_update as incremental_update
from optax._src.update import periodic_update as periodic_update
from optax._src.utils import multi_normal as multi_normal
from optax._src.utils import scale_gradient as scale_gradient
from optax._src.utils import value_and_grad_from_state as value_and_grad_from_state

# TODO(mtthss): remove contrib aliases from flat namespace once users updated.
# Deprecated modules
from optax.contrib import differentially_private_aggregate as _deprecated_differentially_private_aggregate
from optax.contrib import DifferentiallyPrivateAggregateState as _deprecated_DifferentiallyPrivateAggregateState
from optax.contrib import dpsgd as _deprecated_dpsgd


# TODO(mtthss): remove aliases after updates.
adaptive_grad_clip = transforms.adaptive_grad_clip
AdaptiveGradClipState = EmptyState
clip = transforms.clip
clip_by_block_rms = transforms.clip_by_block_rms
clip_by_global_norm = transforms.clip_by_global_norm
ClipByGlobalNormState = EmptyState
ClipState = EmptyState
per_example_global_norm_clip = transforms.per_example_global_norm_clip
per_example_layer_norm_clip = transforms.per_example_layer_norm_clip
keep_params_nonnegative = transforms.keep_params_nonnegative
NonNegativeParamsState = transforms.NonNegativeParamsState
zero_nans = transforms.zero_nans
ZeroNansState = transforms.ZeroNansState
chain = transforms.chain
multi_transform = transforms.partition
MultiTransformState = transforms.PartitionState
named_chain = transforms.named_chain
trace = transforms.trace
TraceState = transforms.TraceState
ema = transforms.ema
EmaState = transforms.EmaState
add_noise = transforms.add_noise
AddNoiseState = transforms.AddNoiseState
add_decayed_weights = transforms.add_decayed_weights
AddDecayedWeightsState = EmptyState
ScaleByTrustRatioState = EmptyState
ScaleState = EmptyState
apply_if_finite = transforms.apply_if_finite
ApplyIfFiniteState = transforms.ApplyIfFiniteState
conditionally_mask = transforms.conditionally_mask
conditionally_transform = transforms.conditionally_transform
ConditionallyMaskState = transforms.ConditionallyMaskState
ConditionallyTransformState = transforms.ConditionallyTransformState
flatten = transforms.flatten
masked = transforms.masked
MaskedNode = transforms.MaskedNode
MaskedState = transforms.MaskedState
MultiSteps = transforms.MultiSteps
MultiStepsState = transforms.MultiStepsState
ShouldSkipUpdateFunction = transforms.ShouldSkipUpdateFunction
skip_large_updates = transforms.skip_large_updates
skip_not_finite = transforms.skip_not_finite

# TODO(mtthss): remove tree_utils aliases after updates.
tree_map_params = tree_utils.tree_map_params
bias_correction = tree_utils.tree_bias_correction
update_infinity_moment = tree_utils.tree_update_infinity_moment
update_moment = tree_utils.tree_update_moment
update_moment_per_elem_norm = tree_utils.tree_update_moment_per_elem_norm

# TODO(mtthss): remove schedules aliases from flat namespaces after user updates
constant_schedule = schedules.constant_schedule
cosine_decay_schedule = schedules.cosine_decay_schedule
cosine_onecycle_schedule = schedules.cosine_onecycle_schedule
exponential_decay = schedules.exponential_decay
inject_hyperparams = schedules.inject_hyperparams
InjectHyperparamsState = schedules.InjectHyperparamsState
join_schedules = schedules.join_schedules
linear_onecycle_schedule = schedules.linear_onecycle_schedule
linear_schedule = schedules.linear_schedule
piecewise_constant_schedule = schedules.piecewise_constant_schedule
piecewise_interpolate_schedule = schedules.piecewise_interpolate_schedule
polynomial_schedule = schedules.polynomial_schedule
sgdr_schedule = schedules.sgdr_schedule
warmup_constant_schedule = schedules.warmup_constant_schedule
warmup_cosine_decay_schedule = schedules.warmup_cosine_decay_schedule
warmup_exponential_decay_schedule = schedules.warmup_exponential_decay_schedule
inject_stateful_hyperparams = schedules.inject_stateful_hyperparams
InjectStatefulHyperparamsState = schedules.InjectStatefulHyperparamsState
WrappedSchedule = schedules.WrappedSchedule

# TODO(mtthss): remove loss aliases from flat namespace once users have updated.
convex_kl_divergence = losses.convex_kl_divergence
cosine_distance = losses.cosine_distance
cosine_similarity = losses.cosine_similarity
ctc_loss = losses.ctc_loss
ctc_loss_with_forward_probs = losses.ctc_loss_with_forward_probs
hinge_loss = losses.hinge_loss
huber_loss = losses.huber_loss
kl_divergence = losses.kl_divergence
l2_loss = losses.l2_loss
log_cosh = losses.log_cosh
ntxent = losses.ntxent
sigmoid_binary_cross_entropy = losses.sigmoid_binary_cross_entropy
smooth_labels = losses.smooth_labels
safe_softmax_cross_entropy = losses.safe_softmax_cross_entropy
softmax_cross_entropy = losses.softmax_cross_entropy
softmax_cross_entropy_with_integer_labels = (
    losses.softmax_cross_entropy_with_integer_labels
)
squared_error = losses.squared_error
sigmoid_focal_loss = losses.sigmoid_focal_loss

_deprecations = {
    # Added Apr 2024
    "differentially_private_aggregate": (
        (
            "optax.differentially_private_aggregate is deprecated: use"
            " optax.contrib.differentially_private_aggregate (optax v0.1.8 or"
            " newer)."
        ),
        _deprecated_differentially_private_aggregate,
    ),
    "DifferentiallyPrivateAggregateState": (
        (
            "optax.DifferentiallyPrivateAggregateState is deprecated: use"
            " optax.contrib.DifferentiallyPrivateAggregateState (optax v0.1.8"
            " or newer)."
        ),
        _deprecated_DifferentiallyPrivateAggregateState,
    ),
    "dpsgd": (
        (
            "optax.dpsgd is deprecated: use optax.contrib.dpsgd (optax v0.1.8"
            " or newer)."
        ),
        _deprecated_dpsgd,
    ),
}
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
if _typing.TYPE_CHECKING:
  # pylint: disable=reimported
  from optax.contrib import differentially_private_aggregate as differentially_private_aggregate
  from optax.contrib import DifferentiallyPrivateAggregateState as DifferentiallyPrivateAggregateState
  from optax.contrib import dpsgd as dpsgd
  # pylint: enable=reimported

else:
  from optax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
# pylint: enable=g-importing-member


__version__ = "0.2.5.dev"

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
