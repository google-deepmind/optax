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
"""The losses sub-package."""

from optax.losses.classification import convex_kl_divergence
from optax.losses.classification import ctc_loss
from optax.losses.classification import ctc_loss_with_forward_probs
from optax.losses.classification import hinge_loss
from optax.losses.classification import kl_divergence
from optax.losses.classification import kl_divergence_with_log_targets
from optax.losses.classification import poly_loss_cross_entropy
from optax.losses.classification import sigmoid_binary_cross_entropy
from optax.losses.classification import softmax_cross_entropy
from optax.losses.classification import softmax_cross_entropy_with_integer_labels
from optax.losses.regression import cosine_distance
from optax.losses.regression import cosine_similarity
from optax.losses.regression import huber_loss
from optax.losses.regression import l2_loss
from optax.losses.regression import log_cosh
from optax.losses.regression import squared_error
from optax.losses.smoothing import smooth_labels
from optax.losses.segmentation import sigmoid_focal_loss
