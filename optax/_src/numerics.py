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
"""Utilities to ensure the implementation is safe wrt numerical issues."""

import chex
import jax.numpy as jnp


def safe_norm(x, min_norm):
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

  The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
  because jax will evaluate both branches of the `jnp.maximum`. This function
  will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
  """
  norm = jnp.linalg.norm(x)
  x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
  return jnp.where(norm <= min_norm, min_norm, jnp.linalg.norm(x))


def safe_root_mean_squares(x, min_rms):
  """Returns jnp.maximum(jnp.sqrt(jnp.mean(x**2)), min_norm) with correct grads.

  The gradients of `jnp.maximum(jnp.sqrt(jnp.mean(x**2)), min_norm)` at 0.0
  is `NaN`, because jax will evaluate both branches of the `jnp.maximum`. This
  function will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_rms: lower bound for the returned norm.
  """
  rms = jnp.sqrt(jnp.mean(x ** 2))
  x = jnp.where(rms <= min_rms, jnp.ones_like(x), x)
  return jnp.where(rms <= min_rms, min_rms, jnp.sqrt(jnp.mean(x ** 2)))


def safe_int32_increment(count):
  """Increments int32 counter by one.

  Normally `max_int + 1` would overflow to `min_int`. This functions ensures
  that when `max_int` is reached the counter stays at `max_int`.

  Args:
    count: a counter to be incremented.

  Returns:
    a counter incremented by 1, or max_int if the maximum precision is reached.
  """
  chex.assert_type(count, jnp.int32)
  max_int32_value = jnp.iinfo(jnp.int32).max
  one = jnp.array(1, dtype=jnp.int32)
  return jnp.where(count < max_int32_value, count + one, max_int32_value)

