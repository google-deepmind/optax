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
"""Shared Hutchinson estimator utilities."""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from optax._src import base
import optax.tree


class HutchinsonState(NamedTuple):
  key: jax.Array


def hutchinson_estimator_diag_hessian(
    random_seed: Optional[jax.Array] = None,
    n_samples: int = 1,
):
  """Returns a GradientTransformationExtraArgs computing the Hessian diagonal.

  The Hessian diagonal is estimated using Hutchinson's estimator, which is
  unbiased but has high variance. Multiple samples reduce variance.

  Args:
    random_seed: key used to generate random vectors.
    n_samples: number of Hutchinson samples to average over per update.

  Returns:
    GradientTransformationExtraArgs
  """
  if n_samples < 1:
    raise ValueError("n_samples must be >= 1.")

  def init_fn(params):
    del params
    # Initialize the RNG key used to draw Rademacher probe vectors.
    key = random_seed if random_seed is not None else jax.random.PRNGKey(0)
    return HutchinsonState(key=key)

  def update_fn(updates, state, params=None, obj_fn=None, **extra_args):
    # Comply with GradientTransformationExtraArgs signature; ignore extra args.
    del extra_args, updates
    if params is None:
      raise ValueError("params must be provided to hutchinson update function.")
    if obj_fn is None:
      raise ValueError("obj_fn must be provided to hutchinson update function.")

    # Split the RNG once per Hutchinson probe.
    key, *subkeys = jax.random.split(state.key, n_samples + 1)

    def one_sample(subkey):
      # Draw Rademacher vectors and compute v ⊙ (H v) via a JVP.
      random_signs = optax.tree.random_like(
          subkey,
          params,
          jax.random.rademacher,
          dtype=jnp.float32,
      )
      random_signs = optax.tree.cast(
          random_signs, optax.tree.dtype(params, "lowest")
      )
      hvp = jax.jvp(jax.grad(obj_fn), (params,), (random_signs,))[1]
      return jax.tree.map(lambda h, r: h * r, hvp, random_signs)

    # Average multiple unbiased probes to reduce estimator variance.
    samples = [one_sample(sk) for sk in subkeys]

    def sum_tree(x, y):
      return jax.tree.map(lambda a, b: a + b, x, y)

    hessian_diag = samples[0]
    for sample in samples[1:]:
      hessian_diag = sum_tree(hessian_diag, sample)

    # Normalize by number of samples to keep an unbiased estimator.
    hessian_diag = jax.tree.map(lambda x: x / n_samples, hessian_diag)
    return hessian_diag, HutchinsonState(key=key)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
