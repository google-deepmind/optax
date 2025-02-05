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
"""Contributed optimizers in Optax."""

# pylint: disable=g-importing-member

from optax.contrib._acprop import acprop as acprop
from optax.contrib._acprop import scale_by_acprop as scale_by_acprop
from optax.contrib._ademamix import ademamix as ademamix
from optax.contrib._ademamix import scale_by_ademamix as scale_by_ademamix
from optax.contrib._ademamix import ScaleByAdemamixState as ScaleByAdemamixState
from optax.contrib._cocob import cocob as cocob
from optax.contrib._cocob import COCOBState as COCOBState
from optax.contrib._cocob import scale_by_cocob as scale_by_cocob
from optax.contrib._complex_valued import split_real_and_imaginary as split_real_and_imaginary
from optax.contrib._complex_valued import SplitRealAndImaginaryState as SplitRealAndImaginaryState
from optax.contrib._dadapt_adamw import dadapt_adamw as dadapt_adamw
from optax.contrib._dadapt_adamw import DAdaptAdamWState as DAdaptAdamWState
from optax.contrib._dog import dog as dog
from optax.contrib._dog import DoGState as DoGState
from optax.contrib._dog import dowg as dowg
from optax.contrib._dog import DoWGState as DoWGState
from optax.contrib._mechanic import MechanicState as MechanicState
from optax.contrib._mechanic import mechanize as mechanize
from optax.contrib._momo import momo as momo
from optax.contrib._momo import momo_adam as momo_adam
from optax.contrib._momo import MomoAdamState as MomoAdamState
from optax.contrib._momo import MomoState as MomoState
from optax.contrib._muon import muon as muon
from optax.contrib._muon import MuonState as MuonState
from optax.contrib._muon import scale_by_muon as scale_by_muon
from optax.contrib._privacy import differentially_private_aggregate as differentially_private_aggregate
from optax.contrib._privacy import DifferentiallyPrivateAggregateState as DifferentiallyPrivateAggregateState
from optax.contrib._privacy import dpsgd as dpsgd
from optax.contrib._prodigy import prodigy as prodigy
from optax.contrib._prodigy import ProdigyState as ProdigyState
from optax.contrib._reduce_on_plateau import reduce_on_plateau as reduce_on_plateau
from optax.contrib._reduce_on_plateau import ReduceLROnPlateauState as ReduceLROnPlateauState
from optax.contrib._sam import normalize as normalize
from optax.contrib._sam import NormalizeState as NormalizeState
from optax.contrib._sam import sam as sam
from optax.contrib._sam import SAMState as SAMState
from optax.contrib._schedule_free import schedule_free as schedule_free
from optax.contrib._schedule_free import schedule_free_adamw as schedule_free_adamw
from optax.contrib._schedule_free import schedule_free_eval_params as schedule_free_eval_params
from optax.contrib._schedule_free import schedule_free_sgd as schedule_free_sgd
from optax.contrib._schedule_free import ScheduleFreeState as ScheduleFreeState
from optax.contrib._sophia import hutchinson_estimator_diag_hessian as hutchinson_estimator_diag_hessian
from optax.contrib._sophia import HutchinsonState as HutchinsonState
from optax.contrib._sophia import sophia as sophia
from optax.contrib._sophia import SophiaState as SophiaState
