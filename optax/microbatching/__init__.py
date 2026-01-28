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
"""Module providing a general `microbatch` transformation."""
from . import _microbatching

Accumulator = _microbatching.Accumulator
AccumulationType = _microbatching.AccumulationType
microbatch = _microbatching.microbatch
micro_grad = _microbatching.micro_grad
micro_vmap = _microbatching.micro_vmap
reshape_batch_axis = _microbatching.reshape_batch_axis
