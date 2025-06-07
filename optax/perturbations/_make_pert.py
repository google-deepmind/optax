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
) -> Callable[[chex.PRNGKey, chex.ArrayTree], chex.ArrayTree]:
  r"""Turns a function into a differentiable approximation, with perturbations.

  For a function :math:`f` (``fun``), a differentiable proxy of :math:`f` is its
  smoothing by perturbation, :math:`f_\sigma`, defined by

  .. math::

    f_\sigma(x) = E[f(x +\sigma Z)]

  for :math:`Z` a random variable sample from the noise sampler.

  :func:`optax.make_perturbed_fun` returns a Monte-Carlo estimate of
  :math:`f_\sigma` and any derivative of :math:`f_\sigma`
  computed through jax (like :func:`jax.grad` but higher derivatives are ok too)
  will return a Monte-Carlo estimate of that derivative.

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
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from optax.perturbations import make_perturbed_fun
    >>> key = jax.random.key(0)
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> f = lambda x: jnp.sum(jnp.maximum(x, 0.0))
    >>> fn = make_perturbed_fun(f, 1_000, 0.1)
    >>> with jnp.printoptions(precision=2):
    ...   print(jax.grad(fn, argnums=1)(key, x))
    [0.69 0.72 0.58]

  .. note::
    For the curious reader, the function :math:`f_\sigma` can equivalently be
    written as

    .. math::

      f_\sigma(x) = E_{y\sim p_{x, \sigma}}[f(y)]

    where :math:`p_{x, \sigma}` is the probability density function of the
    random variable x +\sigma Z.

    The gradient can then be obtained by the score function estimator, a.k.a.
    REINFORCE. We implement the score function estimator through the "magic
    box" operator introduced by Foerster et al, 2018, so that the returned
    function provides stochastic estimators of any order derivatives by simply
    using jax auto-diff system.

  References:
    Berthet et al., `Learning with Differentiable Perturbed Optimizers
    <https://arxiv.org/abs/2002.08676>`_, 2020

    Foerster et al., `DiCE: The Infinitely Differentiable Monte Carlo Estimator
    <https://arxiv.org/abs/1802.05098>`_, 2018

  .. seealso::
    * :doc:`../_collections/examples/perturbations` example.
  """

  def stoch_estimator(key: chex.PRNGKey, x: chex.ArrayTree) -> chex.ArrayTree:
    sample = optax.tree.random_like(key, x, sampler=noise.sample)
    shifted_sample = jax.tree.map(lambda x, z: x + sigma * z, x, sample)
    shifted_sample = jax.lax.stop_gradient(shifted_sample)
    sample = jax.tree.map(lambda x, y: (y - x) / sigma, x, shifted_sample)

    log_prob_sample = optax.tree.sum(jax.tree.map(noise.log_prob, sample))
    out = optax.tree.scale(_magicbox(log_prob_sample), fun(shifted_sample))
    return out

  def mc_estimator(key: chex.PRNGKey, x: chex.ArrayTree) -> chex.ArrayTree:
    out = jax.vmap(stoch_estimator, in_axes=(0, None), out_axes=0)(
        jax.random.split(key, num_samples), x
    )
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), out)

  return mc_estimator


def _magicbox(x):
  """MagicBox operator."""
  return jnp.exp(x - jax.lax.stop_gradient(x))
