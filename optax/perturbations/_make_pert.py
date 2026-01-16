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
      sample_shape: base.Shape = (),
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
      sample_shape: base.Shape = (),
      dtype: jax.typing.DTypeLike = float,
  ) -> jax.Array:
    return jax.random.gumbel(key, sample_shape, dtype)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -inputs - jnp.exp(-inputs)


def make_perturbed_fun(
    fun: Callable[[chex.ArrayTree], chex.ArrayTree],
    num_samples: int = 1000,
    sigma: jax.typing.ArrayLike = 0.1,
    noise=Gumbel(),
    use_baseline=True,
) -> Callable[[base.PRNGKey, chex.ArrayTree], chex.ArrayTree]:
  r"""Returns a differentiable approximation of a function, using stochastic perturbations.

  Let :math:`f` be a function, :math:`\sigma` be a scalar, :math:`\mu` be a
  noise distribution, and

  .. math::
    f_\sigma(x) = \mathbb{E}_{z \sim \mu} f(x + \sigma z)

  Given certain conditions on :math:`\mu`, :math:`f_\sigma` is a smoothed,
  differentiable approximation of :math:`f`, even if :math:`f` itself is not
  differentiable.

  :func:`optax.perturbations.make_perturbed_fun` yields a stochastic function
  whose values and arbitrary-order derivatives (when computed through `JAX's
  automatic differentiation
  <https://docs.jax.dev/en/latest/automatic-differentiation.html>`_
  system) are unbiased Monte-Carlo estimates of the corresponding values and
  derivatives of :math:`f_\sigma`. These estimates are computed using only
  values (not derivatives) of :math:`f`, at stochastic perturbations of the
  input. Thus :math:`f` itself does not have to be differentiable.

  Args:
    fun: The function to transform into a differentiable function. The signature
      currently supported is from pytree to pytree, whose leaves are JAX arrays.
    num_samples: an int, the number of perturbed outputs to average over.
    sigma: a float, the scale of the random perturbation.
    noise: a distribution object that implements ``sample`` and ``log_prob``
      methods, like :class:`optax.perturbations.Gumbel` (which is the default).
    use_baseline: Use the value of the function at the unperturbed input as a
      baseline for variance reduction.

  Returns:
    A new function with the same signature as the original function, but with a
    leading random PRNG key argument.

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
    For the curious reader, :math:`f_\sigma` can also be expressed as

    .. math::
      f_\sigma(x) = \mathbb{E}_{y \sim \nu(x, \sigma)} f(y)

    where :math:`\nu(x, \sigma)` is the probability distribution of the random
    variable :math:`x + \sigma z`.

    The gradient can then be obtained by the score function estimator, a.k.a.
    REINFORCE. We implement the score function estimator through the "magic
    box" operator introduced by Foerster et al, 2018, so that the returned
    function provides stochastic estimates of derivatives of any order by simply
    using `JAX's automatic differentiation
    <https://docs.jax.dev/en/latest/automatic-differentiation.html>`_ system.

  References:
    Berthet et al., `Learning with Differentiable Perturbed Optimizers
    <https://arxiv.org/abs/2002.08676>`_, 2020

    Foerster et al., `DiCE: The Infinitely Differentiable Monte Carlo Estimator
    <https://arxiv.org/abs/1802.05098>`_, 2018

    Salimans et al., `Evolution Strategies as a Scalable Alternative to
    Reinforcement Learning <https://arxiv.org/abs/1703.03864>`_, 2017

  .. seealso::
    * :doc:`../_collections/examples/perturbations` example.
  """  # noqa: E501

  def mc_estimator(key: base.PRNGKey, x: chex.ArrayTree) -> chex.ArrayTree:

    def stoch_estimator(
        key: base.PRNGKey, x: chex.ArrayTree, baseline: chex.ArrayTree
    ) -> chex.ArrayTree:
      sample = optax.tree.random_like(key, x, sampler=noise.sample)
      shifted_sample = jax.tree.map(lambda x, z: x + sigma * z, x, sample)
      shifted_sample = jax.lax.stop_gradient(shifted_sample)
      sample = jax.tree.map(lambda x, y: (y - x) / sigma, x, shifted_sample)

      log_prob_sample = optax.tree.sum(jax.tree.map(noise.log_prob, sample))
      box = _magicbox(log_prob_sample)
      out = optax.tree.scale(box, fun(shifted_sample))
      if use_baseline:
        out = optax.tree.add_scale(out, 1 - box, baseline)
      return out

    if use_baseline:
      baseline = fun(jax.lax.stop_gradient(x))
    else:
      baseline = None

    out = jax.vmap(stoch_estimator, in_axes=(0, None, None), out_axes=0)(
        jax.random.split(key, num_samples), x, baseline
    )
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), out)

  return mc_estimator


def _magicbox(x):
  """MagicBox operator."""
  return jnp.exp(x - jax.lax.stop_gradient(x))
