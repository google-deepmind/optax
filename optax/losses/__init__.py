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

# pylint:disable=g-importing-member

from optax.losses._classification import convex_kl_divergence as convex_kl_divergence
from optax.losses._classification import ctc_loss as ctc_loss
from optax.losses._classification import ctc_loss_with_forward_probs as ctc_loss_with_forward_probs
from optax.losses._classification import hinge_loss as hinge_loss
from optax.losses._classification import kl_divergence as kl_divergence
from optax.losses._classification import kl_divergence_with_log_targets as kl_divergence_with_log_targets
from optax.losses._classification import multiclass_hinge_loss as multiclass_hinge_loss
from optax.losses._classification import multiclass_perceptron_loss as multiclass_perceptron_loss
from optax.losses._classification import multiclass_sparsemax_loss as multiclass_sparsemax_loss
from optax.losses._classification import perceptron_loss as perceptron_loss
from optax.losses._classification import poly_loss_cross_entropy as poly_loss_cross_entropy
from optax.losses._classification import safe_softmax_cross_entropy as safe_softmax_cross_entropy
from optax.losses._classification import sigmoid_binary_cross_entropy as sigmoid_binary_cross_entropy
from optax.losses._classification import sigmoid_focal_loss as sigmoid_focal_loss
from optax.losses._classification import softmax_cross_entropy as softmax_cross_entropy
from optax.losses._classification import softmax_cross_entropy_with_integer_labels as softmax_cross_entropy_with_integer_labels
from optax.losses._classification import sparsemax_loss as sparsemax_loss
from optax.losses._fenchel_young import make_fenchel_young_loss as make_fenchel_young_loss
from optax.losses._ranking import ranking_softmax_loss as ranking_softmax_loss
from optax.losses._regression import cosine_distance as cosine_distance
from optax.losses._regression import cosine_similarity as cosine_similarity
from optax.losses._regression import huber_loss as huber_loss
from optax.losses._regression import l2_loss as l2_loss
from optax.losses._regression import log_cosh as log_cosh
from optax.losses._regression import squared_error as squared_error
from optax.losses._self_supervised import ntxent as ntxent
from optax.losses._self_supervised import triplet_margin_loss as triplet_margin_loss
from optax.losses._smoothing import smooth_labels as smooth_labels
