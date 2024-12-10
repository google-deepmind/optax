"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


from typing import Any, List, NamedTuple, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils


class MuonState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates


def scale_by_muon(
    newton_schulz_coeffs: Union[
        Tuple[float, float, float],
        List[Tuple[float, float, float]],
    ] = (3.4445, -4.7750, 2.0315),
    newton_schulz_steps: Optional[int] = 5,
    mumentum: float = 0.95,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  References:
    Jordan, `Overview of mini-batch gradient descent
    https://github.com/KellerJordan/modded-nanogpt`_, 2024

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    newton_schulz_coeffs: Coefficients for the Newton-schulz method.
    newton_schulz_steps: Number of Newton-schulz iterations.
    mumentum: Exponential decay rate to track the first moment of past
      gradients.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    A `GradientTransformation` object.
  """
  muon_coeffs = jnp.asarray(
      newton_schulz_coeffs
      if isinstance(newton_schulz_coeffs, list)
      else [newton_schulz_coeffs] * newton_schulz_steps
  )
  muon_iterator = (
    lambda x, abc: (abc[0]*x + abc[1]*(x@x.T)@x + abc[2]*(x@x.T)@(x@x.T)@x, 0)
  )
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    return MuonState(count=jnp.zeros([], jnp.int32), mu=mu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, mumentum, 1)
    count_inc = numerics.safe_int32_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
        lambda m, g: mumentum * m + (1 - mumentum) * g,
        otu.tree_bias_correction(
          mu, mumentum, numerics.safe_int32_increment(count_inc)
        ),
        otu.tree_bias_correction(updates, mumentum, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, mumentum, count_inc)
    updates = jax.tree.map(
      lambda x: (
        x / jnp.linalg.norm(x, ord='fro')
        if len(x.shape) > 1
        else x / jnp.linalg.norm(x, ord=2)
      ),
      mu_hat,
    )
    updates, _ = jax.lax.scan(muon_iterator, updates, muon_coeffs)
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, MuonState(count=count_inc, mu=mu)
  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    newton_schulz_coeffs: Union[
        Tuple[float, float, float],
        List[Tuple[float, float, float]],
    ] = (3.4445, -4.7750, 2.0315),
    newton_schulz_steps: Optional[int] = 5,
    mumentum: float = 0.95,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = True,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  References:
    Jordan, `Overview of mini-batch gradient descent
    https://github.com/KellerJordan/modded-nanogpt`_, 2024

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    newton_schulz_coeffs: Coefficients for the Newton-schulz method.
    newton_schulz_steps: Number of Newton-schulz iterations.
    mumentum: Exponential decay rate to track the first moment of past
      gradients.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      scale_by_muon(
          newton_schulz_coeffs,
          newton_schulz_steps,
          mumentum,
          mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
