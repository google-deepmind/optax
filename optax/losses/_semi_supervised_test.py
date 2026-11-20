# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for semi-supervised losses in `optax.losses._semi_supervised.py`."""
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

from optax.losses import _semi_supervised


def _assert_allclose(got, expected, dtype):
  got = np.asarray(got, dtype=np.float32)
  expected = np.asarray(expected, dtype=np.float32)
  if dtype == jnp.bfloat16:
    atol, rtol = 3e-2, 5e-3
  else:
    atol, rtol = 1e-5, 1e-5
  np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


def _assert_finite(x):
  x = np.asarray(x, dtype=np.float32)
  if not np.all(np.isfinite(x)):
    raise AssertionError(f"Expected finite values, got {x}")


class FixMatchLossTest(parameterized.TestCase):
  @parameterized.parameters([
      {
          "seed": 0,
          "B": 4,
          "U": 8,
          "C": 5,
          "soft_labels": False,
          "confidence_threshold": 0.95,
          "lambda_u": 1.0,
          "dtype": jnp.float32,
      },
      {
          "seed": 1,
          "B": 2,
          "U": 6,
          "C": 3,
          "soft_labels": True,
          "confidence_threshold": 0.80,
          "lambda_u": 0.5,
          "dtype": jnp.float32,
      },
      {
          "seed": 2,
          "B": 3,
          "U": 0,
          "C": 4,
          "soft_labels": False,
          "confidence_threshold": 0.90,
          "lambda_u": 1.0,
          "dtype": jnp.float32,
      },
      {
          "seed": 3,
          "B": 3,
          "U": 5,
          "C": 6,
          "soft_labels": False,
          "confidence_threshold": 0.95,
          "lambda_u": 1.0,
          "dtype": jnp.bfloat16,
      },
  ])
  def test_random_batched_matches_reference(
      self, seed, B, U, C, soft_labels, confidence_threshold, lambda_u, dtype
  ):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    labeled_logits = (jax.random.normal(k1, (B, C), dtype=jnp.float32)
                       * 2.0).astype(dtype)
    uw = (jax.random.normal(k2, (U, C), dtype=jnp.float32) * 2.0).astype(dtype)
    us = (jax.random.normal(k3, (U, C), dtype=jnp.float32) * 2.0).astype(dtype)

    # Labels: either int [B] or soft [B,C] (kept float32 for stability).
    if soft_labels:
      raw = jax.random.uniform(k4, (B, C), minval=0.0, maxval=1.0,
                                dtype=jnp.float32)
      labeled_labels = raw / jnp.sum(raw, axis=-1, keepdims=True)
    else:
      labeled_labels = jax.random.randint(k4, (B,), 0, C, dtype=jnp.int32)

    def mean_ce(logits, labels):
      logits = jnp.asarray(logits)
      labels = jnp.asarray(labels)
      if labels.ndim == 1:
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
          logits, labels))
      return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    def reference(ll, lab, uw_, us_, tau, lam):
      # reference in float32 to avoid bf16 rounding mismatches.
      ll = jnp.asarray(ll, dtype=jnp.float32)
      uw_ = jnp.asarray(uw_, dtype=jnp.float32)
      us_ = jnp.asarray(us_, dtype=jnp.float32)

      sup = mean_ce(ll, lab)

      def with_unlabeled(_):
        p_w = jax.nn.softmax(uw_, axis=-1)
        max_p = jnp.max(p_w, axis=-1)
        pseudo = jnp.argmax(p_w, axis=-1)
        mask = (max_p >= jnp.asarray(tau,
                                     dtype=max_p.dtype)).astype(jnp.float32)

        per_ex = optax.softmax_cross_entropy_with_integer_labels(us_, pseudo)
        denom = jnp.asarray(jnp.maximum(us_.shape[0], 1), dtype=jnp.float32)
        unsup = jnp.sum(mask * per_ex) / denom
        return sup + jnp.asarray(lam, dtype=jnp.float32) * unsup

      return lax.cond(us_.shape[0] == 0,
                       lambda _: sup, with_unlabeled, operand=None)

    expected = jax.jit(reference)(
        labeled_logits, labeled_labels, uw, us, confidence_threshold, lambda_u
    )

    got = jax.jit(_semi_supervised.fixmatch_loss)(
        labeled_logits,
        labeled_labels,
        uw,
        us,
        confidence_threshold=confidence_threshold,
        lambda_u=lambda_u,
    )

    with self.subTest("allclose"):
      _assert_allclose(got, expected, dtype)
    with self.subTest("finite"):
      _assert_finite(got)

  @parameterized.parameters([
      {
          "seed": 10, "N": 3, "B": 4, "U": 5, "C": 6,
          "confidence_threshold": 0.0, "lambda_u": 1.0, "dtype": jnp.float32,
      },
      {
          "seed": 11, "N": 2, "B": 2, "U": 3, "C": 4,
          "confidence_threshold": 0.9, "lambda_u": 0.5, "dtype": jnp.float32,
      },
      {
          "seed": 12, "N": 2, "B": 3, "U": 3, "C": 5,
          "confidence_threshold": 0.0, "lambda_u": 1.0, "dtype": jnp.bfloat16,
      },
  ])
  def test_random_vmap(self, seed, N, B, U, C,
                        confidence_threshold, lambda_u, dtype):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    labeled_logits = (jax.random.normal(k1, (N, B, C), dtype=jnp.float32) *
                       2.0).astype(dtype)
    labeled_labels = jax.random.randint(k2, (N, B), 0, C, dtype=jnp.int32)
    uw = (jax.random.normal(k3, (N, U, C), dtype=jnp.float32) *
           2.0).astype(dtype)
    us = (jax.random.normal(k4, (N, U, C), dtype=jnp.float32) *
           2.0).astype(dtype)

    # "Original" computed via lax.map (JAX loop primitive), not vmap.
    def per_item(args):
      ll, lab, xw, xs = args
      return _semi_supervised.fixmatch_loss(
          ll, lab, xw, xs,
          confidence_threshold=confidence_threshold,
          lambda_u=lambda_u,
      )
    original = lax.map(per_item, (labeled_logits, labeled_labels, uw, us))
    vmap_fn = jax.vmap(
        lambda ll, lab, xw, xs: _semi_supervised.fixmatch_loss(
            ll, lab, xw, xs,
            confidence_threshold=confidence_threshold,
            lambda_u=lambda_u,
        ),
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )

    got = jax.jit(vmap_fn)(labeled_logits, labeled_labels, uw, us)
    with self.subTest("allclose"):
      _assert_allclose(got, original, dtype)
    with self.subTest("finite"):
      _assert_finite(got)

  @parameterized.parameters([
      {"seed": 20, "B": 5, "U": 7, "C": 6, "dtype": jnp.float32},
      {"seed": 21, "B": 3, "U": 3, "C": 4, "dtype": jnp.bfloat16},
  ])
  def test_random_permutation_invariance(self, seed, B, U, C, dtype):
    confidence_threshold = 0.0
    lambda_u = 1.0

    key = jax.random.key(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    labeled_logits = (jax.random.normal(k1, (B, C), dtype=jnp.float32) *
                       2.0).astype(dtype)
    labeled_labels = jax.random.randint(k2, (B,), 0, C, dtype=jnp.int32)
    uw = (jax.random.normal(k3, (U, C), dtype=jnp.float32) * 2.0).astype(dtype)
    us = (jax.random.normal(k4, (U, C), dtype=jnp.float32) * 2.0).astype(dtype)

    loss0 = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=confidence_threshold,
        lambda_u=lambda_u,
    )

    perm_b = jax.random.permutation(k5, B)
    perm_u = jax.random.permutation(k6, U)
    loss1 = _semi_supervised.fixmatch_loss(
        labeled_logits[perm_b],
        labeled_labels[perm_b],
        uw[perm_u], us[perm_u],
        confidence_threshold=confidence_threshold,
        lambda_u=lambda_u,
    )
    with self.subTest("allclose"):
      _assert_allclose(loss0, loss1, dtype)
    with self.subTest("finite_loss0"):
      _assert_finite(loss0)
    with self.subTest("finite_loss1"):
      _assert_finite(loss1)

  @parameterized.parameters([
      {"seed": 30, "B": 3, "U": 5, "C": 6, "magnitude": 100.0},
      {"seed": 31, "B": 2, "U": 3, "C": 4, "magnitude": 200.0},
  ])
  def test_random_numerical_stability_extreme_logits(
    self, seed, B, U, C, magnitude):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def extreme_logits(k, shape):
      x = jax.random.normal(k, shape, dtype=jnp.float32)
      s = jnp.where(x >= 0, 1.0, -1.0)
      return s * jnp.asarray(magnitude, dtype=jnp.float32)

    labeled_logits = extreme_logits(k1, (B, C))
    labeled_labels = jax.random.randint(k2, (B,), 0, C, dtype=jnp.int32)
    uw = extreme_logits(k3, (U, C))
    us = extreme_logits(k4, (U, C))

    loss = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=0.0,
        lambda_u=1.0,
    )
    with self.subTest("finite_loss"):
      _assert_finite(loss)

    def f(ll, strong):
      return _semi_supervised.fixmatch_loss(
          ll, labeled_labels,
          uw, strong,
          confidence_threshold=0.0,
          lambda_u=1.0,
      )

    g_ll, g_us = jax.grad(f, argnums=(0, 1))(labeled_logits, us)
    with self.subTest("finite_grad_labeled_logits"):
      _assert_finite(g_ll)
    with self.subTest("finite_grad_strong_logits"):
      _assert_finite(g_us)

  # deterministic tests
  def test_lambda_u_zero_is_supervised_only(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    uw = jnp.array([[10.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]], dtype=jnp.float32)
    us = jnp.array([[0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0]], dtype=jnp.float32)

    got = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=0.0,
        lambda_u=0.0,
    )
    expected = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
          labeled_logits, labeled_labels)
    )
    with self.subTest("allclose"):
      np.testing.assert_allclose(
          np.asarray(got, np.float32), np.asarray(
            expected, np.float32), atol=1e-6, rtol=1e-6
      )
    with self.subTest("finite"):
      _assert_finite(got)

  def test_confidence_threshold_edges(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    uw = jnp.array([[50.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]], dtype=jnp.float32)
    us = jnp.array([[0.0, 50.0, 0.0],
                    [0.0, 0.0, 50.0]], dtype=jnp.float32)

    sup = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
      labeled_logits, labeled_labels))

    got_none = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=1.1,
        lambda_u=1.0,
    )
    with self.subTest("none_allclose_sup"):
      np.testing.assert_allclose(
          np.asarray(got_none, np.float32),
            np.asarray(sup, np.float32), atol=1e-6, rtol=1e-6
      )
    with self.subTest("none_finite"):
      _assert_finite(got_none)

    got_all = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=0.0,
        lambda_u=1.0,
    )
    with self.subTest("all_ge_sup"):
      self.assertGreaterEqual(float(got_all), float(sup) - 1e-6)
    with self.subTest("all_finite"):
      _assert_finite(got_all)

  def test_empty_unlabeled_batch_is_supervised_only(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    uw = jnp.zeros((0, 3), dtype=jnp.float32)
    us = jnp.zeros((0, 3), dtype=jnp.float32)

    got = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=0.0,
        lambda_u=1.0,
    )
    expected = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
          labeled_logits, labeled_labels)
    )
    with self.subTest("allclose"):
      np.testing.assert_allclose(
          np.asarray(got, np.float32), np.asarray(
            expected, np.float32), atol=1e-6, rtol=1e-6
      )
    with self.subTest("finite"):
      _assert_finite(got)

  def test_grad_flows_through_strong_logits(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    uw = jnp.array([[50.0, 0.0, 0.0],
                    [50.0, 0.0, 0.0]], dtype=jnp.float32)
    us = jnp.array([[0.0, 50.0, 0.0],
                    [0.0, 0.0, 50.0]], dtype=jnp.float32)

    def loss_wrt_strong(strong):
      return _semi_supervised.fixmatch_loss(
          labeled_logits,
          labeled_labels, uw,
          strong,
          confidence_threshold=0.0,
          lambda_u=1.0,
      )

    g_us = jax.grad(loss_wrt_strong)(us)
    with self.subTest("finite_grad"):
      _assert_finite(g_us)
    with self.subTest("nonzero_grad"):
      self.assertTrue(bool(jnp.any(jnp.abs(g_us) > 1e-8)))

  def test_bfloat16_runs(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.bfloat16)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    uw = jnp.array([[10.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]], dtype=jnp.bfloat16)
    us = jnp.array([[0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0]], dtype=jnp.bfloat16)

    loss = _semi_supervised.fixmatch_loss(
        labeled_logits,
        labeled_labels, uw, us,
        confidence_threshold=0.95,
        lambda_u=1.0,
    )
    with self.subTest("finite"):
      _assert_finite(loss)


class MixMatchLossTest(parameterized.TestCase):
  @parameterized.parameters([
      {"seed": 100, "B": 4, "U": 8, "C": 5, "soft_labels": False,
        "lambda_u": 10.0, "dtype": jnp.float32},
      {"seed": 101, "B": 2, "U": 6, "C": 3, "soft_labels": True,
        "lambda_u": 5.0, "dtype": jnp.float32},
      {"seed": 102, "B": 3, "U": 5, "C": 6, "soft_labels": False,
        "lambda_u": 10.0, "dtype": jnp.bfloat16},
  ])
  def test_random_batched_matches_reference(
    self, seed, B, U, C, soft_labels, lambda_u, dtype):
    key = jax.random.key(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    labeled_logits = (jax.random.normal(k1, (B, C), dtype=jnp.float32) *
                       2.0).astype(dtype)
    unlabeled_logits = (jax.random.normal(k2, (U, C), dtype=jnp.float32) *
                         2.0).astype(dtype)

    raw_u = jax.random.uniform(k3, (U, C), minval=0.0,
                                maxval=1.0, dtype=jnp.float32)
    unlabeled_targets = raw_u / jnp.sum(raw_u, axis=-1, keepdims=True)

    if soft_labels:
      raw_l = jax.random.uniform(k4, (B, C),
                                  minval=0.0, maxval=1.0, dtype=jnp.float32)
      labeled_labels = raw_l / jnp.sum(raw_l, axis=-1, keepdims=True)
    else:
      labeled_labels = jax.random.randint(k5, (B,), 0, C, dtype=jnp.int32)

    def mean_ce(logits, labels):
      logits = jnp.asarray(logits)
      labels = jnp.asarray(labels)
      if labels.ndim == 1:
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
          logits, labels))
      return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    def reference(ll, lab, ul, ut, lam):
      # Reference computed in float32 to reduce bf16 rounding mismatches.
      ll = jnp.asarray(ll, dtype=jnp.float32)
      ul = jnp.asarray(ul, dtype=jnp.float32)
      ut = jnp.asarray(ut, dtype=jnp.float32)

      sup = mean_ce(ll, lab)
      p = jax.nn.softmax(ul, axis=-1)
      q = lax.stop_gradient(ut)
      unsup = jnp.mean(jnp.sum(optax.squared_error(p, q), axis=-1))
      return sup + jnp.asarray(lam, dtype=jnp.float32) * unsup

    expected = jax.jit(reference)(
      labeled_logits, labeled_labels,
      unlabeled_logits, unlabeled_targets, lambda_u)

    got = jax.jit(_semi_supervised.mixmatch_loss)(
        labeled_logits,
        labeled_labels,
        unlabeled_logits,
        unlabeled_targets,
        lambda_u=lambda_u,
    )
    with self.subTest("allclose"):
      _assert_allclose(got, expected, dtype)
    with self.subTest("finite"):
      _assert_finite(got)

  @parameterized.parameters([
      {"seed": 110, "N": 3, "B": 4, "U": 5, "C": 6,
        "lambda_u": 10.0, "dtype": jnp.float32},
      {"seed": 111, "N": 2, "B": 2, "U": 3, "C": 4,
        "lambda_u": 5.0, "dtype": jnp.float32},
      {"seed": 112, "N": 2, "B": 3, "U": 4, "C": 5,
        "lambda_u": 10.0, "dtype": jnp.bfloat16},
  ])
  def test_random_vmap(self, seed, N, B, U, C, lambda_u, dtype):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    labeled_logits = (jax.random.normal(k1, (N, B, C), dtype=jnp.float32) *
                       2.0).astype(dtype)
    labeled_labels = jax.random.randint(k2, (N, B), 0, C, dtype=jnp.int32)
    unlabeled_logits = (jax.random.normal(k3, (N, U, C), dtype=jnp.float32) *
                         2.0).astype(dtype)

    raw = jax.random.uniform(k4, (N, U, C), minval=0.0,
                              maxval=1.0, dtype=jnp.float32)
    unlabeled_targets = raw / jnp.sum(raw, axis=-1, keepdims=True)

    # "Original" computed via lax.map (JAX loop primitive), not vmap.
    def per_item(args):
      ll, lab, ul, ut = args
      return _semi_supervised.mixmatch_loss(ll, lab, ul, ut, lambda_u=lambda_u)

    original = lax.map(per_item, (
      labeled_logits, labeled_labels, unlabeled_logits, unlabeled_targets))

    vmap_fn = jax.vmap(
        lambda ll, lab, ul, ut: _semi_supervised.mixmatch_loss(
          ll, lab, ul, ut, lambda_u=lambda_u),
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )

    got = jax.jit(vmap_fn)(
      labeled_logits, labeled_labels, unlabeled_logits, unlabeled_targets)
    with self.subTest("allclose"):
      _assert_allclose(got, original, dtype)
    with self.subTest("finite"):
      _assert_finite(got)

  @parameterized.parameters([
      {"seed": 120, "B": 5, "U": 7, "C": 6, "dtype": jnp.float32},
      {"seed": 121, "B": 3, "U": 5, "C": 4, "dtype": jnp.bfloat16},
  ])
  def test_random_permutation_invariance(self, seed, B, U, C, dtype):
    lambda_u = 10.0

    key = jax.random.key(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    labeled_logits = (jax.random.normal(k1, (B, C),
                                         dtype=jnp.float32) * 2.0).astype(dtype)
    labeled_labels = jax.random.randint(k2, (B,), 0, C, dtype=jnp.int32)
    unlabeled_logits = (
      jax.random.normal(k3, (U, C), dtype=jnp.float32) * 2.0).astype(dtype)

    raw_u = jax.random.uniform(k4, (U, C),
                                minval=0.0, maxval=1.0, dtype=jnp.float32)
    unlabeled_targets = raw_u / jnp.sum(raw_u, axis=-1, keepdims=True)

    loss0 = _semi_supervised.mixmatch_loss(
        labeled_logits, labeled_labels, unlabeled_logits,
          unlabeled_targets, lambda_u=lambda_u
    )
    perm_b = jax.random.permutation(k5, B)
    perm_u = jax.random.permutation(k6, U)
    loss1 = _semi_supervised.mixmatch_loss(
        labeled_logits[perm_b],
        labeled_labels[perm_b],
        unlabeled_logits[perm_u],
        unlabeled_targets[perm_u],
        lambda_u=lambda_u,
    )
    with self.subTest("allclose"):
      _assert_allclose(loss0, loss1, dtype)
    with self.subTest("finite_loss0"):
      _assert_finite(loss0)
    with self.subTest("finite_loss1"):
      _assert_finite(loss1)

  @parameterized.parameters([
      {"seed": 130, "B": 3, "U": 5, "C": 6, "magnitude": 100.0},
      {"seed": 131, "B": 2, "U": 3, "C": 4, "magnitude": 200.0},
  ])
  def test_random_numerical_stability_extreme_logits(
    self, seed, B, U, C, magnitude):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def extreme_logits(k, shape):
      x = jax.random.normal(k, shape, dtype=jnp.float32)
      s = jnp.where(x >= 0, 1.0, -1.0)
      return s * jnp.asarray(magnitude, dtype=jnp.float32)

    labeled_logits = extreme_logits(k1, (B, C))
    labeled_labels = jax.random.randint(k2, (B,), 0, C, dtype=jnp.int32)
    unlabeled_logits = extreme_logits(k3, (U, C))

    raw_u = jax.random.uniform(
      k4, (U, C), minval=0.0, maxval=1.0, dtype=jnp.float32)
    unlabeled_targets = raw_u / jnp.sum(raw_u, axis=-1, keepdims=True)

    loss = _semi_supervised.mixmatch_loss(
        labeled_logits, labeled_labels,
          unlabeled_logits, unlabeled_targets, lambda_u=10.0
    )
    with self.subTest("finite_loss"):
      _assert_finite(loss)

    def f(ll, ul):
      return _semi_supervised.mixmatch_loss(
        ll, labeled_labels, ul, unlabeled_targets, lambda_u=10.0)

    g_ll, g_ul = jax.grad(f, argnums=(0, 1))(labeled_logits, unlabeled_logits)
    with self.subTest("finite_grad_labeled_logits"):
      _assert_finite(g_ll)
    with self.subTest("finite_grad_unlabeled_logits"):
      _assert_finite(g_ul)

  def test_lambda_u_zero_is_supervised_only(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    unlabeled_logits = jnp.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], dtype=jnp.float32)
    unlabeled_targets = jnp.array([[0.7, 0.2, 0.1],
                                   [0.1, 0.8, 0.1]], dtype=jnp.float32)

    got = _semi_supervised.mixmatch_loss(
        labeled_logits, labeled_labels,
          unlabeled_logits, unlabeled_targets, lambda_u=0.0
    )
    expected = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
          labeled_logits, labeled_labels)
    )
    with self.subTest("allclose"):
      np.testing.assert_allclose(
          np.asarray(got, np.float32), np.asarray(
            expected, np.float32), atol=1e-6, rtol=1e-6
      )
    with self.subTest("finite"):
      _assert_finite(got)

  def test_stop_gradient_unlabeled_targets(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    unlabeled_logits = jnp.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], dtype=jnp.float32)
    unlabeled_targets = jnp.array([[0.7, 0.2, 0.1],
                                   [0.1, 0.8, 0.1]], dtype=jnp.float32)

    def loss_wrt_targets(t):
      return _semi_supervised.mixmatch_loss(
          labeled_logits,
          labeled_labels,
          unlabeled_logits, t,
          lambda_u=10.0,
      )

    g_t = jax.grad(loss_wrt_targets)(unlabeled_targets)
    with self.subTest("allclose_zero_grad"):
      np.testing.assert_allclose(
          np.asarray(g_t, np.float32),
          np.asarray(jnp.zeros_like(g_t), np.float32),
          atol=1e-7,
          rtol=0.0,
      )
    with self.subTest("finite_grad"):
      _assert_finite(g_t)

  def test_unsup_zero_when_targets_match_probs(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.float32)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)

    targets = jnp.array([[0.7, 0.2, 0.1],
                         [0.1, 0.8, 0.1]], dtype=jnp.float32)
    unlabeled_logits = jnp.log(targets + 1e-8)

    got = _semi_supervised.mixmatch_loss(
        labeled_logits, labeled_labels, unlabeled_logits, targets, lambda_u=10.0
    )
    sup = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
      labeled_logits, labeled_labels))
    with self.subTest("allclose_sup"):
      np.testing.assert_allclose(
          np.asarray(got, np.float32),
            np.asarray(sup, np.float32), atol=2e-5, rtol=2e-5
      )
    with self.subTest("finite"):
      _assert_finite(got)

  def test_bfloat16_runs(self):
    labeled_logits = jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=jnp.bfloat16)
    labeled_labels = jnp.array([0, 1], dtype=jnp.int32)
    unlabeled_logits = jnp.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], dtype=jnp.bfloat16)
    unlabeled_targets = jnp.array([[0.7, 0.2, 0.1],
                                   [0.1, 0.8, 0.1]], dtype=jnp.bfloat16)

    loss = _semi_supervised.mixmatch_loss(
        labeled_logits, labeled_labels,
          unlabeled_logits, unlabeled_targets, lambda_u=10.0
    )
    with self.subTest("finite"):
      _assert_finite(loss)


if __name__ == "__main__":
  absltest.main()
