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

"""Creates a differentiable approximation of a function with perturbations."""


import operator
from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp
from optax._src import base
import optax.tree
import functools


class Normal:
  """Normal distribution."""

  def sample(
      self,
      key: jax.typing.ArrayLike,
      sample_shape: base.Shape,
      dtype: jax.typing.DTypeLike = float,
  ) -> jax.Array:
    return jax.random.normal(key, sample_shape, dtype)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -0.5 * inputs**2


class Gumbel:
  """Gumbel distribution."""

  def sample(
      self,
      key: jax.typing.ArrayLike,
      sample_shape: base.Shape,
      dtype: jax.typing.DTypeLike = float,
  ) -> jax.Array:
    return jax.random.gumbel(key, sample_shape, dtype)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -inputs - jnp.exp(-inputs)


def make_perturbed_fun(
    fun: Callable[[chex.ArrayTree], chex.ArrayTree],
    num_samples: int = 1000,
    sigma: float = 0.1,
    noise=Gumbel(),
) -> Callable[[chex.ArrayTree, chex.PRNGKey], chex.ArrayTree]:
  r"""Turns a function into a differentiable approximation, with perturbations.

  For a function :math:`f` (``fun``), it creates a proxy :math:`f_\sigma`
  defined by

  .. math::

    f_\sigma(x) = E[f(x +\sigma Z)]

  for :math:`Z` a random variable sample from the noise sampler. This implements
  a Monte-Carlo estimate.

  Args:
    fun: The function to transform into a differentiable function. Signature
      currently supported is from pytree to pytree, whose leaves are jax arrays.
    num_samples: an int, the number of perturbed outputs to average over.
    sigma: a float, the scale of the random perturbation.
    noise: a distribution object that must implement a sample function and a
      log-pdf of the desired distribution, similar to
      :class:optax.perturbations.Gumbel. Default is Gumbel distribution.

  Returns:
    A function with the same signature (plus an additional rng in the input) so
    that it and its first derivative are implemented as Monte-Carlo estimates
    over values of fun only, not the derivative of fun. The result is therefore
    suitable to differentiate even if fun is not.

    Order n derivatives of the result are evaluated as Monte-Carlo estimates
    over the the order n-1 derivatives of fun. It is therefore suitable to take
    n derivatives of the result if fun is n-1 times differentiable.

  References:
    Berthet et al., `Learning with Differentiable Perturbed Optimizers
    <https://arxiv.org/abs/2002.08676>`_, 2020

  .. seealso::
    * :doc:`../_collections/examples/perturbations` example.
  """

  def _compute_residuals(
      primal_in: chex.ArrayTree, rng: chex.PRNGKey
  ) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    # random noise Zs to be added to inputs
    samples = optax.tree.random_like(rng,
        optax.tree.batch_shape(primal_in, (num_samples,)), sampler=noise.sample)

    # creates [inputs + Z_1, ..., inputs + Z_num_samples]
    primal_in_samples = jax.vmap(lambda z: optax.tree.add_scale(primal_in, sigma, z))(samples)

    # applies fun: [fun(inputs + Z_1), ..., fun(inputs + Z_num_samples)]
    primal_out_samples = jax.vmap(fun)(primal_in_samples)
    return primal_out_samples, samples

  @functools.partial(jax.custom_jvp,nondiff_argnums=(1,))
  def fun_perturb(primal_in: chex.ArrayTree, rng: chex.PRNGKey) -> chex.ArrayTree:
    primal_out_samples, _ = _compute_residuals(primal_in, rng)
    # average the perturbed outputs
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), primal_out_samples)

  @fun_perturb.defjvp
  def fun_perturb_jvp(
          rng: chex.PRNGKey, primal_in: chex.ArrayTree, tangent_in: chex.ArrayTree
  ) -> chex.ArrayTree:
    """Computes the jacobian vector product.

    Following the method in [Berthet et al. 2020], for a vector `g`, we have
    Jac(fun_perturb)(inputs) * g =
    -1/sigma * E[fun(inputs + sigma * Z) * <grad log_prob(Z), g>].
    This implements a Monte-Carlo estimate

    Args:
      tangent: the tangent in the jacobian vector product.
      _: not used.
      inputs: the inputs to the function.
      rng: the random number generator key.

    Returns:
      The jacobian vector product.
    """
    assert len(primal_in) == 1 and len(tangent_in) == 1
    primal_in = primal_in[0]
    tangent_in = tangent_in[0]

    outputs_pert, samples = _compute_residuals(primal_in, rng)
    primal_out = jax.tree.map(lambda x: jnp.mean(x, axis=0), outputs_pert)

    tree_log_prob_func = lambda sample: optax.tree.sum(jax.tree.map(noise.log_prob, sample))
    @functools.partial(jax.vmap,in_axes=(0,0))
    def sample_tangent_out(output_pert, sample):
      _, grad_inner_prod = jax.jvp(tree_log_prob_func, (sample,), (tangent_in,))
      return jax.tree.map(lambda x: x * grad_inner_prod, output_pert)

    tangent_out_samples = sample_tangent_out(outputs_pert, samples)
    tangent_out = jax.tree.map(lambda x: -1/sigma * jnp.mean(x, axis=0), tangent_out_samples)

    return primal_out, tangent_out

  return fun_perturb
