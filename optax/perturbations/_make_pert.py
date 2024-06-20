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

"""Utilities to perform maths on pytrees."""


from collections.abc import Callable
import operator
from typing import Sequence

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base


Shape = base.Shape


class Normal:
  """Normal distribution."""

  def sample(self,
             seed: chex.PRNGKey,
             sample_shape: Shape) -> jax.Array:
    return jax.random.normal(seed, sample_shape)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -0.5 * inputs ** 2


class Gumbel:
  """Gumbel distribution."""

  def sample(self,
             seed: chex.PRNGKey,
             sample_shape: Shape) -> jax.Array:
    return jax.random.gumbel(seed, sample_shape)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -inputs - jnp.exp(-inputs)


def _tree_mean_across(trees: Sequence[chex.ArrayTree]) -> chex.ArrayTree:
  """Mean across a list of trees.

  Args:
    trees: List or tuple of pytrees with the same structure.

  Returns:
    a pytree with the same structure as each tree in ``trees`` with leaves being
    the mean across the trees.

  >>> optax.tree_utils.tree_reduce_mean_across(
  ...   [{'first': [1, 2], 'last': 3},
  ...    {'first': [5, 6], 'last': 7}]
  ...   )
  {'first': [6, 8], 'last': 10}
  """
  mean_fun = lambda x: sum(x) / len(trees)
  return jtu.tree_map(lambda *leaves: mean_fun(leaves), *trees)


def _tree_vmap(fun: Callable[[chex.ArrayTree], chex.ArrayTree],
               trees: Sequence[chex.ArrayTree]) -> chex.ArrayTree:
  """Applies a function to a list of trees, akin to a vmap."""
  tree_def_in = jtu.tree_structure(trees[0])
  has_in_structure = lambda x: jtu.tree_structure(x) == tree_def_in
  return jtu.tree_map(fun, trees, is_leaf=has_in_structure)


def make_perturbed_fun(fun: Callable[[chex.ArrayTree], chex.ArrayTree],
                       num_samples: int = 1000,
                       sigma: float = 0.1,
                       noise=Gumbel()):
  """Turns a function into a differentiable approximation, with perturbations.
  
  This is an implementation of the method described in [Berthet et al., 2020].
  For a function `fun`, it creates a proxy `fun_perturb` defined by
  fun_perturb(inputs) = E[fun(inputs + sigma * Z)],
  where Z is sampled from the `noise` sampler. This implements a Monte-Carlo
  estimate.
  
  References:
    [Berthet et al., 2020]: Learning with Differentiable Perturbed Optimizers
    Q. Berthet, M. Blondel, O. Teboul, M. Cuturi, J-P. Vert, F. Bach
    NeurIPS 2020
  
  Args:
    fun: The function to transform into a differentiable function.
      Signature currently supported is from pytree to pytree, whose leaves are
      jax arrays.
    num_samples: an int, the number of perturbed outputs to average over.
    sigma: a float, the scale of the random perturbation.
    noise: a distribution object that must implement a sample function and a
      log-pdf of the desired distribution, similar to :class: Gumbel.
      Default is Gumbel distribution.

  Returns:
    A function with the same signature (and an rng) that can be automatically
    differentiated.
    
  .. warning: The vjp or gradient of the transformed function does not allow
  additional derivations.
  """

  def _compute_residuals(inputs, rng):
    # random noise Zs to be added to inputs
    samples = [
        otu.tree_random_like(rng_, inputs, sampler=noise.sample)
        for rng_ in jax.random.split(rng, num_samples)
    ]
    # creates [inputs + Z_1, ..., inputs + Z_num_samples]
    inputs_pert = _tree_vmap(
        lambda z: otu.tree_add_scalar_mul(inputs, sigma, z), samples
    )
    # applies fun: [fun(inputs + Z_1), ..., fun(inputs + Z_num_samples)]
    outputs_pert = _tree_vmap(fun, inputs_pert)
    return outputs_pert, samples

  @jax.custom_vjp
  def fun_perturb(inputs, rng):
    outputs_pert, _ = _compute_residuals(inputs, rng)
    # averages the perturbed outputs
    return _tree_mean_across(outputs_pert)

  def fun_perturb_fwd(inputs, rng):
    outputs_pert, samples = _compute_residuals(inputs, rng)
    return _tree_mean_across(outputs_pert), (outputs_pert, samples)

  def fun_perturb_bwd(res, g):
    """Computes the vector jacobian product.
    
    Following the method in [Berthet et al. 2020], for a vector `g`, we have
    g * Jac(fun_perturb)(inputs) = 
    E[g * fun(inputs + sigma * Z) * grad log_prob(Z)].
    This implements a Monte-Carlo estimate
    
    Args:
      res: the residuals computed in `compute_all_residuals`
      g: the vector in the vector jacobian product
      
    Returns:
      The vector jacobian product.
    """
    outputs_pert, samples = res
    # computes a list of scalar products [<g, fun(inputs) + sigma * Z_1>, ...]
    tree_dots = _tree_vmap(lambda t: jtu.tree_map(jnp.dot, t, g),
                           outputs_pert)
    list_dots = _tree_vmap(lambda x: jtu.tree_reduce(operator.add, x),
                           tree_dots)
    # creates a function that mapping Z' to
    # (1 / num_samples) * sum_i <g, fun(inputs) + sigma * Z_i> * log_prob(Z'_i)
    def inner_sum_fun(z):
      # individual log_probs for all leaves
      sum_log_prob_func = lambda x: jnp.sum(noise.log_prob(x))
      tree_sum_log_probs = jtu.tree_map(sum_log_prob_func, z)
      # sums the log_probs by independance.
      list_log_probs = _tree_vmap(lambda x: jtu.tree_reduce(operator.add, x),
                                  tree_sum_log_probs)
      # computes element-wise product with list of scalar products
      list_prods = jtu.tree_map(lambda x, y: x * y, list_dots, list_log_probs)
      return jtu.tree_reduce(operator.add, list_prods) / num_samples
    # computes the vjp by summing the gradient of this function.
    inner_sum_grad_fun = jax.grad(inner_sum_fun)
    grad_final = _tree_mean_across(inner_sum_grad_fun(samples))
    return (grad_final, None)

  fun_perturb.defvjp(fun_perturb_fwd, fun_perturb_bwd)

  return fun_perturb
