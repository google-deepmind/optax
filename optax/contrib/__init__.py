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

from optax.contrib.cocob import cocob
from optax.contrib.cocob import COCOBState
from optax.contrib.complex_valued import split_real_and_imaginary
from optax.contrib.complex_valued import SplitRealAndImaginaryState
from optax.contrib.dadapt_adamw import dadapt_adamw
from optax.contrib.dadapt_adamw import DAdaptAdamWState
from optax.contrib.mechanic import MechanicState
from optax.contrib.mechanic import mechanize
from optax.contrib.privacy import differentially_private_aggregate
from optax.contrib.privacy import DifferentiallyPrivateAggregateState
from optax.contrib.privacy import dpsgd
from optax.contrib.prodigy import prodigy
from optax.contrib.prodigy import ProdigyState
from optax.contrib.reduce_on_plateau import reduce_on_plateau
from optax.contrib.reduce_on_plateau import ReduceLROnPlateauState
from optax.contrib.sam import normalize
from optax.contrib.sam import NormalizeState
from optax.contrib.sam import sam
from optax.contrib.sam import SAMState
