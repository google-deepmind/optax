# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""semi supervised losses."""
import jax
from jax import lax
import jax.numpy as jnp

from optax._src import utils
from optax.losses import _classification
from optax.losses import _regression


def fixmatch_loss(
    labeled_logits: jax.typing.ArrayLike,
    labeled_labels: jax.typing.ArrayLike,
    unlabeled_weak_logits: jax.typing.ArrayLike,
    unlabeled_strong_logits: jax.typing.ArrayLike,
    confidence_threshold: jax.typing.ArrayLike = 0.95,
    lambda_u: jax.typing.ArrayLike = 1.0,
) -> jax.Array:
  """FixMatch loss: supervised CE + thresholded pseudo-label CE (weak -> strong).

  Args:
    labeled_logits: Logits for labeled batch, shape [B, C].
    labeled_labels: Integer labels [B] or soft/one-hot labels [B, C].
    unlabeled_weak_logits: Weak-aug logits for unlabeled batch, shape [U, C].
    unlabeled_strong_logits: Strong-aug logits for unlabeled batch, shape [U, C].
    confidence_threshold: tau; keep pseudo-label if max prob >= tau.
    lambda_u: Weight for unlabeled term.

  Returns:
    Scalar FixMatch loss.
  """
  labeled_logits = jnp.asarray(labeled_logits)
  labeled_labels = jnp.asarray(labeled_labels)
  unlabeled_weak_logits = jnp.asarray(unlabeled_weak_logits)
  unlabeled_strong_logits = jnp.asarray(unlabeled_strong_logits)

  utils.check_subdtype(labeled_logits, jnp.floating)
  utils.check_subdtype(unlabeled_weak_logits, jnp.floating)
  utils.check_subdtype(unlabeled_strong_logits, jnp.floating)

  if labeled_logits.ndim != 2:
    raise ValueError(f'labeled_logits must be rank-2 [B,C], got {labeled_logits.shape}')
  if unlabeled_weak_logits.shape != unlabeled_strong_logits.shape:
    raise ValueError(
        'unlabeled_weak_logits and unlabeled_strong_logits must have the same shape, '
        f'got {unlabeled_weak_logits.shape} vs {unlabeled_strong_logits.shape}'
    )
  if labeled_logits.shape[-1] != unlabeled_weak_logits.shape[-1]:
    raise ValueError(
        'Class dimension must match for labeled/unlabeled logits, got '
        f'{labeled_logits.shape[-1]} vs {unlabeled_weak_logits.shape[-1]}'
    )

  # Supervised CE on labeled batch (supports int or soft labels)
  if labeled_labels.ndim == 1:
    sup = jnp.mean(
        _classification.softmax_cross_entropy_with_integer_labels(
            logits=labeled_logits, labels=labeled_labels
        )
    )
  else:
    sup = jnp.mean(
        _classification.softmax_cross_entropy(
            logits=labeled_logits, labels=labeled_labels
        )
    )

  # Pseudo-label from weak predictions
  probs_w = jax.nn.softmax(unlabeled_weak_logits, axis=-1)  # [U,C]
  max_probs = jnp.max(probs_w, axis=-1)   # [U]
  pseudo = jnp.argmax(probs_w, axis=-1)  # [U]

  tau = jnp.asarray(confidence_threshold, dtype=max_probs.dtype)
  mask = (max_probs >= tau).astype(labeled_logits.dtype)       # [U]

  # Unsupervised CE on strong predictions vs pseudo-labels, masked
  unsup_per_ex = _classification.softmax_cross_entropy_with_integer_labels(
      logits=unlabeled_strong_logits,
      labels=lax.stop_gradient(pseudo),
  )                                                            # [U]

  # Paper-style normalization: divide by U (mu * B)
  denom = jnp.maximum(unlabeled_strong_logits.shape[0], 1)
  unsup = jnp.sum(mask * unsup_per_ex) / denom

  lam = jnp.asarray(lambda_u, dtype=sup.dtype)
  return sup + lam * unsup


def mixmatch_loss(
    labeled_logits: jax.typing.ArrayLike,
    labeled_labels: jax.typing.ArrayLike,
    unlabeled_logits: jax.typing.ArrayLike,
    unlabeled_targets: jax.typing.ArrayLike,
    lambda_u: jax.typing.ArrayLike = 100.0,
) -> jax.Array:
  """MixMatch loss: supervised CE + unlabeled L2/Brier on probabilities.

  This assumes you already performed MixMatch preprocessing (guess labels,
  sharpen, mixup), so:
    - labeled_labels is often soft [B,C] (but [B] int is OK),
    - unlabeled_targets is soft q' with shape [U,C].

  Args:
    labeled_logits: Logits for labeled mixed batch X', shape [B, C].
    labeled_labels: Labels for X' as int [B] or soft [B, C].
    unlabeled_logits: Logits for unlabeled mixed batch U', shape [U, C].
    unlabeled_targets: Soft targets q' for U', shape [U, C].
    lambda_u: Weight for unlabeled term.

  Returns:
    Scalar MixMatch loss.
  """
  labeled_logits = jnp.asarray(labeled_logits)
  labeled_labels = jnp.asarray(labeled_labels)
  unlabeled_logits = jnp.asarray(unlabeled_logits)
  unlabeled_targets = jnp.asarray(unlabeled_targets)

  utils.check_subdtype(labeled_logits, jnp.floating)
  utils.check_subdtype(unlabeled_logits, jnp.floating)
  utils.check_subdtype(unlabeled_targets, jnp.floating)

  if labeled_logits.ndim != 2 or unlabeled_logits.ndim != 2:
    raise ValueError('logits must be rank-2 [*,C].')
  if labeled_logits.shape[-1] != unlabeled_logits.shape[-1]:
    raise ValueError('Class dimension must match between labeled and unlabeled logits.')
  if unlabeled_targets.shape != unlabeled_logits.shape:
    raise ValueError(
        'unlabeled_targets must match unlabeled_logits shape [U,C], got '
        f'{unlabeled_targets.shape} vs {unlabeled_logits.shape}'
    )

  # Supervised CE (supports int or soft labels)
  if labeled_labels.ndim == 1:
    lx = jnp.mean(
        _classification.softmax_cross_entropy_with_integer_labels(
            logits=labeled_logits, labels=labeled_labels
        )
    )
  else:
    lx = jnp.mean(
        _classification.softmax_cross_entropy(
            logits=labeled_logits, labels=labeled_labels
        )
    )

  # Unlabeled term: || q' - softmax(logits) ||^2 (Brier / L2 on probs)
  p = jax.nn.softmax(unlabeled_logits, axis=-1)  # [U,C]
  q = lax.stop_gradient(unlabeled_targets)   # [U,C]
  se = _regression.squared_error(p, q)  # [U,C]
  lu = jnp.mean(jnp.sum(se, axis=-1))   # scalar

  lam = jnp.asarray(lambda_u, dtype=lx.dtype)
  return lx + lam * lu