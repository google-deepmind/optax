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
"""Tests for second order oracles from `optax.second_order._oracles.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
from jax import flatten_util
import jax.numpy as jnp
import jax.random as jrd
import jax.tree_util as jtu
from optax.second_order import _oracles as oracles


def random_split_like_tree(rng_key, target_tree=None, treedef=None):
  """Split keys to match structure of target tree or of tree_def."""
  if treedef is None:
    treedef = jtu.tree_structure(target_tree)
  keys = jrd.split(rng_key, treedef.num_leaves)
  return jtu.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target_tree):
  """Create tree with normal random entries of the same shape as target tree."""
  keys_tree = random_split_like_tree(rng_key, target_tree)
  return jtu.tree_map(
      lambda l, k: jrd.normal(k, l.shape, l.dtype),
      target_tree,
      keys_tree,
  )


def get_random_sym_matrix(key, dim):
  mat = jrd.normal(key, (dim, dim))
  mat = mat + mat.transpose()
  return mat


class MLP(nn.Module):
  num_outputs: int
  hidden_sizes: list[int]

  @nn.compact
  def __call__(self, x):
    for num_hidden in self.hidden_sizes:
      x = nn.Dense(num_hidden)(x)
      x = nn.gelu(x)
    return nn.Dense(self.num_outputs)(x)


def setup_mlp(input_dim, output_dim, hidden_sizes, key):
  mlp = MLP(num_outputs=output_dim, hidden_sizes=hidden_sizes)
  key, key_params, key_input, key_output = jrd.split(key, 4)
  x = jrd.normal(key_input, (input_dim,))
  y = jrd.normal(key_output, (output_dim,))
  params = mlp.init(key_params, jnp.ones(input_dim))
  return mlp, params, x, y, key


class OraclesTest(chex.TestCase):
  """Tests for second order oracles from `optax.second_order._oracles.py`."""

  @chex.all_variants
  def test_hvp_mlp_basic(self):
    # Setup problem
    """Test hvp on mlp."""
    key = jrd.PRNGKey(0)
    input_dim, output_dim = 4, 2
    hidden_sizes = [4, 4]
    mlp, params, x, _, key = setup_mlp(input_dim, output_dim, hidden_sizes, key)
    tangents = tree_random_normal_like(key, params)

    def fn(params):
      z = mlp.apply(params, x)
      return jnp.sum(z**2)

    # Get reference quantities by flattening params
    params_flat, unravel = flatten_util.ravel_pytree(params)
    tangents_flat, _ = flatten_util.ravel_pytree(tangents)

    def fn_flat(params_flat):
      params = unravel(params_flat)
      return fn(params)

    value, grad = jax.value_and_grad(fn)(params)
    hessian = jax.hessian(fn_flat)(params_flat)
    hvp = hessian.dot(tangents_flat)
    hvp = unravel(hvp)

    # Computations via make_hvp_fn
    make_hvp_fn_ = functools.partial(oracles.make_hvp_fn, fn=fn)
    make_hvp_fn = self.variant(make_hvp_fn_)
    value1, grad1, hvp_fn = make_hvp_fn(params=params)
    hvp1 = hvp_fn(tangents)

    # Computations via hvp_call
    hvp_call_ = functools.partial(oracles.hvp_call, fn=fn)
    hvp_call = self.variant(hvp_call_)
    value2, grad2, hvp2 = hvp_call(params=params, tangents=tangents)

    # Check everything
    # If on tpu or gpu matrix multiplications are done at half-precision
    # so we should test at that precision.
    if jax.default_backend() in ['gpu', 'tpu']:
      atol = 10*jnp.finfo('bfloat16').eps
      rtol = 10*jnp.finfo('bfloat16').eps
    else:
      atol = 100*jnp.finfo('float32').eps
      rtol = 100*jnp.finfo('float32').eps
    chex.assert_trees_all_close(value, value1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(value2, value2, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(grad, grad1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(grad1, grad2, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(hvp, hvp1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(hvp1, hvp2, atol=atol, rtol=rtol)

  @chex.all_variants
  def test_hvp_mlp_with_aux_and_fn_kwargs(self):
    """Test hvp on mlp."""
    key = jrd.PRNGKey(0)
    input_dim, output_dim = 4, 2
    hidden_sizes = [4, 4]
    mlp, params, x, y, key = setup_mlp(input_dim, output_dim, hidden_sizes, key)
    tangents = tree_random_normal_like(key, params)

    def fn(params, x, y):
      z = mlp.apply(params, x)
      return jnp.sum((z - y) ** 2), z

    # Get reference quantities by flattening params
    params_flat, unravel = flatten_util.ravel_pytree(params)
    tangents_flat, _ = flatten_util.ravel_pytree(tangents)

    def fn_flat(params_flat, x, y):
      params = unravel(params_flat)
      return fn(params, x, y)

    (value, aux), grad = jax.value_and_grad(fn, has_aux=True)(params, x, y)
    hessian = jax.hessian(fn_flat, has_aux=True)(params_flat, x, y)[0]
    hvp = hessian.dot(tangents_flat)
    hvp = unravel(hvp)

    # Computations via make_hvp_fn
    make_hvp_fn_ = functools.partial(oracles.make_hvp_fn, fn=fn, has_aux=True)
    make_hvp_fn = self.variant(make_hvp_fn_)
    (value1, aux1), grad1, hvp_fn = make_hvp_fn(params=params, x=x, y=y)
    hvp1 = hvp_fn(tangents)

    # Computations via hvp_call
    hvp_call_ = functools.partial(oracles.hvp_call, fn=fn, has_aux=True)
    hvp_call = self.variant(hvp_call_)
    (value2, aux2), grad2, hvp2 = hvp_call(
        params=params, tangents=tangents, x=x, y=y
    )

    # Check everything
    # If on tpu or gpu matrix multiplications are done at half-precision
    # so we should test at that precision.
    if jax.default_backend() in ['gpu', 'tpu']:
      atol = 10*jnp.finfo('bfloat16').eps
      rtol = 10*jnp.finfo('bfloat16').eps
    else:
      atol = 100*jnp.finfo('float32').eps
      rtol = 100*jnp.finfo('float32').eps
    chex.assert_trees_all_close(value, value1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(value2, value2, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(aux, aux1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(aux1, aux2, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(grad, grad1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(grad1, grad2, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(hvp, hvp1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(hvp1, hvp2, atol=atol, rtol=rtol)

  @parameterized.product(
      inner_has_aux=[True, False], outer_has_aux=[True, False]
  )
  def test_gnvp_mlp_with_outer_aux_and_fn_kwargs(
      self, inner_has_aux, outer_has_aux
  ):
    # Setup problem
    key = jrd.PRNGKey(0)
    input_dim, output_dim = 4, 2
    hidden_sizes = [4, 4]
    mlp, params, x, y, key = setup_mlp(input_dim, output_dim, hidden_sizes, key)
    params_flat, unravel = flatten_util.ravel_pytree(params)
    tangents = tree_random_normal_like(key, params)
    tangents_flat, _ = flatten_util.ravel_pytree(tangents)

    # Define inner and outer functions with or without aux
    if inner_has_aux:

      def inner_fn(params, x):
        return mlp.apply(params, x), x

    else:

      def inner_fn(params, x):
        return mlp.apply(params, x)

    if outer_has_aux:

      def outer_fn(outputs, y):
        return jnp.sum((outputs - y) ** 2), outputs

    else:

      def outer_fn(outputs, y):
        return jnp.sum((outputs - y) ** 2)

    # Define composition to get reference value and grad
    def fn(params, x, y):
      outputs_and_maybe_aux = inner_fn(params, x)
      if inner_has_aux:
        outputs, inner_aux = outputs_and_maybe_aux
      else:
        outputs = outputs_and_maybe_aux
        inner_aux = None

      value_and_maybe_aux = outer_fn(outputs, y)
      if outer_has_aux:
        value, outer_aux = value_and_maybe_aux
      else:
        value = value_and_maybe_aux
        outer_aux = None
      return value, (inner_aux, outer_aux)

    # Get reference values (by flattening params for the gnvp)
    (value, (inner_aux, outer_aux)), grad = jax.value_and_grad(
        fn, has_aux=True
    )(params, x, y)

    def inner_fn_flat(params_flat, x_flat):
      params = unravel(params_flat)
      if inner_has_aux:
        return inner_fn(params, x_flat)[0]
      else:
        return inner_fn(params, x_flat)

    jac = jax.jacobian(inner_fn_flat)(params_flat, x)
    outputs = inner_fn_flat(params_flat, x)
    hessian_and_maybe_aux = jax.hessian(outer_fn, has_aux=outer_has_aux)(
        outputs, y
    )
    if outer_has_aux:
      hessian = hessian_and_maybe_aux[0]
    else:
      hessian = hessian_and_maybe_aux
    gnvp_flat = jac.T.dot(hessian.dot(jac.dot(tangents_flat)))
    gnvp = unravel(gnvp_flat)

    # Computed by gnvp
    make_gnvp_fn = functools.partial(
        oracles.make_gnvp_fn,
        inner_fn=inner_fn,
        outer_fn=outer_fn,
        inner_fn_has_aux=inner_has_aux,
        outer_fn_has_aux=outer_has_aux,
    )
    value_and_maybe_aux, grad1, gnvp_fn = make_gnvp_fn(
        params=params,
        x=x,
        y=y,
    )
    gnvp1 = gnvp_fn(tangents)
    if inner_has_aux or outer_has_aux:
      value1, (inner_aux1, outer_aux1) = value_and_maybe_aux
    else:
      value1 = value_and_maybe_aux
      inner_aux1 = None
      outer_aux1 = None

    # Check everything
    # If on tpu or gpu matrix multiplications are done at half-precision
    # so we should test at that precision.
    if jax.default_backend() in ['gpu', 'tpu']:
      atol = 10*jnp.finfo('bfloat16').eps
      rtol = 10*jnp.finfo('bfloat16').eps
    else:
      atol = 100*jnp.finfo('float32').eps
      rtol = 100*jnp.finfo('float32').eps
    chex.assert_trees_all_close(value, value1, atol=atol, rtol=rtol)
    if inner_has_aux or outer_has_aux:
      chex.assert_trees_all_close(inner_aux, inner_aux1, atol=atol, rtol=rtol)
      chex.assert_trees_all_close(outer_aux, outer_aux1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(grad, grad1, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(gnvp, gnvp1, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
