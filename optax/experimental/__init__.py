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
"""Experimental features in Optax.

Features may be removed or modified at any time.
"""

# TODO(mtthss): delete import stubs after user updates.
from optax import contrib
from optax.experimental.extra_args import GradientTransformationWithExtraArgs
from optax.experimental.extra_args import named_chain

# TODO(mtthss): delete import stubs after user updates.
split_real_and_imaginary = contrib.split_real_and_imaginary
SplitRealAndImaginaryState = contrib.SplitRealAndImaginaryState
del contrib
