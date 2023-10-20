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
"""Utilities for creating schedules."""

from optax.schedules.inject import inject_hyperparams
from optax.schedules.inject import InjectHyperparamsState
from optax.schedules.join import join_schedules
from optax.schedules.schedule import constant_schedule
from optax.schedules.schedule import cosine_decay_schedule
from optax.schedules.schedule import cosine_onecycle_schedule
from optax.schedules.schedule import exponential_decay
from optax.schedules.schedule import linear_onecycle_schedule
from optax.schedules.schedule import linear_schedule
from optax.schedules.schedule import piecewise_constant_schedule
from optax.schedules.schedule import piecewise_interpolate_schedule
from optax.schedules.schedule import polynomial_schedule
from optax.schedules.schedule import sgdr_schedule
from optax.schedules.schedule import warmup_cosine_decay_schedule
from optax.schedules.schedule import warmup_exponential_decay_schedule
from optax.schedules.stateful import inject_stateful_hyperparams
from optax.schedules.stateful import InjectStatefulHyperparamsState
from optax.schedules.stateful import WrappedSchedule
