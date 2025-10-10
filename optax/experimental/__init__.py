# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Experimental features for Optax."""

from optax.experimental.aggregate import add_mean_variance_to_opt
from optax.experimental.aggregate import Aggregator
from optax.experimental.aggregate import average_incrementally_updates
from optax.experimental.aggregate import get_unbiased_mean_and_variance_ema
from optax.experimental.aggregate import process
