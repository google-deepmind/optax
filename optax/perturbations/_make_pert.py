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


from typing import Callable

import chex
import jax
import jax.numpy as jnp
from optax._src import base
import optax.tree


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
    A function with the same signature (preceded by a random key in the input)
    so that it and its first derivative are implemented as Monte-Carlo estimates
    over values of fun only, not the derivative of fun. The result is therefore
    suitable to differentiate even if fun is not.

  Example:
    >>> key = jax.random.key(0)
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> f = lambda x: jnp.sum(jnp.maximum(x, 0.0))
    >>> fp = _make_pert.make_perturbed_fun(f, 1_000, 0.1)
    >>> with jnp.printoptions(precision=2):
    ...   print(jax.grad(fp, argnums=1)(key, x))
    [0.63 0.59 0.63]

  References:
    Berthet et al., `Learning with Differentiable Perturbed Optimizers
    <https://arxiv.org/abs/2002.08676>`_, 2020

  .. seealso::
    * :doc:`../_collections/examples/perturbations` example.
  """

  def _compute_residuals(key: chex.PRNGKey, primal_in: chex.ArrayTree
                         ) -> tuple[chex.ArrayTree, chex.ArrayTree]:

    if not all(isinstance(x, jax.Array) for x in jax.tree.leaves(primal_in)):
      raise ValueError('All leaves of primal_in must be jax.Array, got: '
                       f'{jax.tree.map(type, primal_in)}')

    # construct sample using optax.tree.random_like with added batch dimension
    # but without explicitly realizing the batched `_like` primals.
    samples = optax.tree.random_like(  # random noise Zs to be added to inputs
        key, jax.tree.map(lambda x: jax.ShapeDtypeStruct(
            (num_samples,) + x.shape, x.dtype), primal_in),
        sampler=noise.sample)

    # applies fun: [fun(inputs + Z_1), ..., fun(inputs + Z_num_samples)]
    primal_out_samples = jax.vmap(
        lambda sample: fun(optax.tree.add_scale(primal_in, sigma, sample)))(
            samples)
    return primal_out_samples, samples

  @jax.custom_jvp
  def fun_perturb(key: chex.PRNGKey,
                  primal_in: chex.ArrayTree) -> chex.ArrayTree:
    primal_out_samples, _ = _compute_residuals(key, primal_in)
    # average the perturbed outputs
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), primal_out_samples)

  @fun_perturb.defjvp  # pytype: disable=wrong-arg-types
  def _fun_perturb_jvp(primal_in: chex.ArrayTree,
                       tangent_in: chex.ArrayTree) -> chex.ArrayTree:
    """Computes the jacobian vector product.

    Following the method in [Berthet et al. 2020], for a vector `g`, we have
    Jac(fun_perturb)(inputs) * g =
    -1/sigma * E[fun(inputs + sigma * Z) * <grad log_prob(Z), g>].
    This implements a Monte-Carlo estimate

    Args:
      primal_in: forwad function inputs
      tangent_in: the tangent in the jacobian vector product.

    Returns:
      The jacobian vector product.
    """
    assert len(primal_in) == 2 and len(tangent_in) == 2
    key, primal_in = primal_in
    key_tangent_in, tangent_in = tangent_in
    del key_tangent_in

    outputs_pert, samples = _compute_residuals(key, primal_in)
    primal_out = jax.tree.map(lambda x: jnp.mean(x, axis=0), outputs_pert)

    tree_log_prob_func = lambda sample: optax.tree.sum(
        jax.tree.map(noise.log_prob, sample))

    broadcast_tangent_in = optax.tree.batch_shape(tangent_in, (num_samples,))
    _, grad_inner_prod = jax.jvp(
        jax.vmap(tree_log_prob_func), (samples,), (broadcast_tangent_in,))

    tangent_out_samples = jax.tree.map(
        lambda x: x * jnp.expand_dims(grad_inner_prod, axis=range(1, x.ndim)),
        outputs_pert)

    tangent_out = jax.tree.map(lambda x: -1 / sigma * jnp.mean(x, axis=0),
                               tangent_out_samples)

    return primal_out, tangent_out

  return fun_perturb
