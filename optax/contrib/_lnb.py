"""Linear Neuron Boosting.

We refer to a "Neuron" as a callable PyTree that is linear in its parameters.
The input array to a Neuron must have shape (d_in, ...); this axis assumption
is noted in the code via "AA".
"""

import functools
import itertools
from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeAlias, Union

import jax
import jax.numpy as jnp
import jax.scipy
import jax.tree_util as jtu
import optax
from optax._src import base
from optax._src import combine
from optax._src import utils
import optax.tree_utils as otu

Neuron: TypeAlias = base.PyTree
"""A Callable PyTree that is linear in its parameters (weights and biases)."""

Weight: TypeAlias = jax.Array
"""A Neuron's weight array with expected shape=(d_out, d_in, ...)."""

Bias: TypeAlias = jax.Array
"""A Neuron's bias array with expected shape=(d_out,) or (d_out, 1, ..., 1)."""

IsNeuron: TypeAlias = Callable[[base.PyTree], bool]
"""Returns True if the node is a Neuron; callled during tree flattening."""

IsWeight: TypeAlias = Callable[[jtu.KeyPath], bool]
"""Called on a Neuron's KeyPath, returns True if it is for the weights."""

IsBias: TypeAlias = Callable[[jtu.KeyPath], bool]
"""Called on a Neuron's KeyPath, returns True if it is for the biases."""

Vector: TypeAlias = jax.Array
"""A 1-D array."""


_zip = functools.partial(zip, strict=True)


def _get_array(
    neuron: Neuron, predicate: Union[IsWeight, IsBias]
) -> Optional[jax.Array]:
    """Returns the array for the leaf where the predicate evaluates True."""
    leaves, _ = jtu.tree_flatten_with_path(neuron)
    for key_path, arr in leaves:
        if predicate(key_path):
            return arr
    return None


def _set_array(
    neuron: Neuron, predicate: Union[IsWeight, IsBias], new_array: jax.Array
) -> Neuron:
    """Assigns `new_array` for the leaf where the predicate evaluates True."""

    def _replace(key_path: jtu.KeyPath, curr_array: jax.Array) -> jax.Array:
        return new_array if predicate(key_path) else curr_array

    return jtu.tree_map_with_path(_replace, neuron)


def _last_name_is_str(key_path: jtu.KeyPath, query: str) -> bool:
    """Returns True if the last EntryName in `key_path` matches `query`."""
    last_entry = key_path[-1]
    entry_name = getattr(last_entry, "key", getattr(last_entry, "name", None))
    return entry_name == query


def _compute_mu(xs: jax.Array) -> Vector:
    """Returns the mean feature vector across the spatial dimensions."""
    return jnp.mean(xs, axis=(0, *tuple(range(2, xs.ndim))))  # AA


def _shrink(diag: Vector, shrinkage: jax.typing.ArrayLike) -> Vector:
    """See https://scikit-learn.org/stable/modules/covariance.html#basic-shrinkage."""
    assert 0.0 <= shrinkage
    assert shrinkage < 1.0
    return (1.0 - shrinkage) * diag + (
        shrinkage * jnp.sum(diag) / jnp.size(diag)
    )


def default_is_weight(key_path: jtu.KeyPath) -> bool:
    """Returns True if the EntryName is "weight"."""
    return _last_name_is_str(key_path, "weight")


def default_is_bias(key_path: jtu.KeyPath) -> bool:
    """Returns True if the EntryName is "bias"."""
    return _last_name_is_str(key_path, "bias")


def make_pvp(
    mu: Vector,
    nu: Vector,
    shrinkage: jax.typing.ArrayLike,
    is_weight: IsWeight,
    is_bias: IsBias,
) -> Callable[[Neuron], Neuron]:
    """Returns the Neuron's preconditioner mvp for conjugate gradient.

    The preconditioner is the incomplete Cholesky factorization (Section 3.5).

    Args:
      mu: mean input features
      nu: mean squared input features
      shrinkage: covariance shinkage in [0, 1)
      is_weight: see type definition
      is_bias: see type definition
    """
    assert len(mu.shape) == 1
    assert mu.shape == nu.shape
    variance_shrunk = _shrink(jnp.maximum(0.0, nu - mu**2), shrinkage)

    # PVP for one component of the parameter vector (with bias).
    def _pvp_bias(
        weight: jax.Array, bias: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        # We use vmap here because `weight` might be an N-D spatial filter and
        # we use scalar broadcasting to equally apply the feature moments over
        # the spatial axes (i.e., assumes translational equivariance).
        weight_out = jax.vmap(
            lambda w, b, m, v: (w - m * b) / v, in_axes=(0, None, 0, 0)
        )(weight, bias, mu, variance_shrunk)
        bias_out = bias - jnp.sum(jax.vmap(lambda m, w: m * w)(mu, weight_out))
        assert weight_out.shape == weight.shape
        assert bias_out.shape == bias.shape
        return weight_out, bias_out

    # PVP for one component of the parameter vector (without bias).
    def _pvp_nobias(weight: jax.Array) -> jax.Array:
        nu_shrunk = variance_shrunk + mu**2
        return jax.vmap(lambda w, n: w / n)(weight, nu_shrunk)

    # PVP for all components of the parameter vector.
    def pvp(v: Neuron) -> Neuron:
        weight = _get_array(v, is_weight)
        assert weight is not None
        assert weight.shape[1] == mu.size  # AA
        bias = _get_array(v, is_bias)
        if bias is None:
            weight = jax.vmap(_pvp_nobias)(weight)
            v = _set_array(v, is_weight, weight)
        else:
            weight, bias = jax.vmap(_pvp_bias)(weight, bias)
            v = _set_array(v, is_weight, weight)
            v = _set_array(v, is_bias, bias)
        return v

    return pvp


def project(
    xs: jax.Array,
    grad: Neuron,
    init: Neuron,
    is_weight: IsWeight,
    is_bias: IsBias,
    ridge: float = 0.0,
    **kwargs: Any,
) -> Neuron:
    """Preconditions neuron's component of the gradient vector.

    Conjugate gradient is used to solve the linear system (Section 3.2).

    Args:
      xs: the batched input tensors to the neuron
      grad: the component of the gradient vector for the neuron
      init: initialization for conjugate gradient
      is_weight: see type definition
      is_bias: see type definition
      ridge: ridge reglarization; only applied to the weights
      kwargs: forwards to jax.scipy.sparse.linalg.cg
    """

    def jvp(v: Neuron) -> jax.Array:
        return jax.vmap(v)(xs)

    ys = jvp(init)
    num_samples = ys.size / ys.shape[1]  # AA
    del ys
    vjp = jax.linear_transpose(
        jvp, init
    )  # The linearization point doesn't matter since linear.

    def mvp(v: Neuron) -> Neuron:
        return otu.tree_scalar_mul(1.0 / num_samples, vjp(jvp(v))[0])  # J'Jv

    # Create a pytree with same shape as the neuron and use scalar broadcasting
    # to implement ridge regression while ignoring bias, if present.
    ridge_neuron = _set_array(init, is_weight, ridge)
    ridge_neuron = _set_array(ridge_neuron, is_bias, 0.0)

    def mvp_ridge(v: Neuron) -> Neuron:
        return otu.tree_add(mvp(v), otu.tree_mul(ridge_neuron, v))

    _mvp = mvp_ridge if ridge > 0.0 else mvp
    return jax.scipy.sparse.linalg.cg(_mvp, grad, x0=init, **kwargs)[0]


class ScaleByLNBState(NamedTuple):
    """State for the Linear Neuron Boosting algorithm."""

    # Each state is a list of length number of neurons, in leaf traversal order
    h_neurons: list[Neuron]  # The previous conjugate gradient solution
    mu_state: optax.EmaState  # Mean input features (first moment)
    nu_state: optax.EmaState  # Mean squared input features (second moment)


def scale_by_lnb(
    b_mu: jax.typing.ArrayLike = 0.9,
    b_nu: jax.typing.ArrayLike = 0.999,
    min_norm: jax.typing.ArrayLike = 1e-2,
    cov_shrinkage: jax.typing.ArrayLike = 0.1,
    *,
    cg_ridge: float = 0.1,
    cg_maxiter: int = 2,
    approx_metric: bool = False,
    is_neuron: IsNeuron = lambda node: hasattr(node, "weight"),
    is_weight: IsWeight = default_is_weight,
    is_bias: IsBias = default_is_bias,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Applies a Linear Neuron Boosting update.

    LNB performs gradient descent in the space of linear functions for each
    linear layer of the network. Equivalently, it is a trust region method that
    takes a small step under a metric related to the second moment matrix of
    each linear layer's input features. Also equivalently, it runs gradient
    descent on a reparameterized model where each linear layer whitens its input
    features.

    Args:
      b_mu: EMA decay for input features' first moment.
      b_nu: EMA decay for input features' second momment.
      min_norm: The minimum norm to use to avoid dividing by zero.
      cov_shrinkage: Covariance shrinkage, in [0, 1), to ensure positive
        definiteness.
      cg_ridge: Ridge regularizer for conjugate gradient.
      cg_maxiter: Number of conjugate gradient iterations.
      approx_metric: Approximate the metric with the incomplete Cholesky
        factorization, instead of using conjugate gradient to solve the linear
        system.
      is_neuron: See type definition.
      is_weight: See type definition.
      is_bias: See type definition.
      accumulator_dtype: EMA accumulator dtype.

    Returns:
      `GradientTransformationExtraArgs` with an `update_fn` that expects the
      `xs_neurons` kwarg to be a list of batched input arrays to each respective
      neuron, in leaf traversal order as specified by `is_neuron`.

    References:
      D. Munoz, `Simple Linear Neuron Boosting
      <https://arxiv.org/abs/2502.01131>`_, 2025
    """
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    mu_ema = optax.ema(b_mu, debias=True, accumulator_dtype=accumulator_dtype)
    nu_ema = optax.ema(b_nu, debias=True, accumulator_dtype=accumulator_dtype)
    _make_pvp = functools.partial(
        make_pvp, shrinkage=cov_shrinkage, is_weight=is_weight, is_bias=is_bias
    )
    _project = functools.partial(
        project,
        is_weight=is_weight,
        is_bias=is_bias,
        ridge=cg_ridge,
        maxiter=cg_maxiter,
    )

    def init_fn(params: base.Params) -> ScaleByLNBState:
        def _make_zero_vector(neuron: Neuron) -> Vector:
            weight = _get_array(neuron, is_weight)
            assert weight is not None
            return jnp.zeros(weight.shape[1])  # AA

        neurons = list(
            filter(is_neuron, jtu.tree_leaves(params, is_leaf=is_neuron))
        )
        assert all(map(callable, neurons)), "Each Neuron must be callable."
        return ScaleByLNBState(
            h_neurons=list(map(otu.tree_zeros_like, neurons)),
            mu_state=mu_ema.init(list(map(_make_zero_vector, neurons))),
            nu_state=nu_ema.init(list(map(_make_zero_vector, neurons))),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleByLNBState,
        params: base.Params,
        xs_neurons: list[jax.Array],
    ) -> Tuple[base.Updates, ScaleByLNBState]:
        del params
        # Update feature moments.
        mu_neurons = list(map(_compute_mu, xs_neurons))
        nu_neurons = list(map(_compute_mu, map(jnp.square, xs_neurons)))
        mu_neurons, mu_state = mu_ema.update(mu_neurons, state.mu_state)
        nu_neurons, nu_state = nu_ema.update(nu_neurons, state.nu_state)

        # Construct PVPs for conjugate gradient using moments.
        pvp_neurons = itertools.starmap(_make_pvp, _zip(mu_neurons, nu_neurons))

        # Grab components of the entire gradient vector that are neurons.
        leaves, treedef = jtu.tree_flatten(updates, is_leaf=is_neuron)
        idx_neurons, grad_neurons = _zip(
            *[(i, node) for i, node in enumerate(leaves) if is_neuron(node)]
        )

        # Precondition the gradient vector and then update its neurons.
        if approx_metric:
            h_neurons = [
                pvp(grad) for grad, pvp in _zip(grad_neurons, pvp_neurons)
            ]
        else:
            h_neurons = [
                _project(xs, grad, init, M=pvp)
                for xs, grad, init, pvp in _zip(
                    xs_neurons,
                    grad_neurons,
                    state.h_neurons,
                    pvp_neurons,
                )
            ]

        # Replace the Neurons (implies the Identity metric for the others).
        for idx, neuron in _zip(idx_neurons, h_neurons):
            leaves[idx] = neuron
        h = jtu.tree_unflatten(treedef, leaves)

        # Rescale under the metric (adaptive step size).
        norm = jnp.maximum(min_norm, jnp.sqrt(otu.tree_vdot(h, updates)))
        h_unit = otu.tree_scalar_mul(1.0 / norm, h)
        return h_unit, ScaleByLNBState(h_neurons, mu_state, nu_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def lnb(
    b_g: jax.typing.ArrayLike = 0.9,
    b_mu: jax.typing.ArrayLike = 0.9,
    b_nu: jax.typing.ArrayLike = 0.999,
    min_norm: jax.typing.ArrayLike = 1e-2,
    cov_shrinkage: jax.typing.ArrayLike = 0.1,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    *,
    cg_ridge: float = 0.1,
    cg_maxiter: int = 2,
    approx_metric: bool = False,
    is_neuron: IsNeuron = lambda node: hasattr(node, "weight"),
    is_weight: IsWeight = default_is_weight,
    is_bias: IsBias = default_is_bias,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Linear Neuron Boosting.

    LNB performs gradient descent in the space of linear functions for each
    linear layer of the network. Equivalently, it is a trust region method that
    takes a small step under a metric related to the second moment matrix of
    each linear layer's input features. Also equivalently, it runs gradient
    descent on a reparameterized model where each linear layer whitens its input
    features.

    Args:
      b_g: EMA decay for the gradient vector.
      b_mu: EMA decay for input features' first moment.
      b_nu: EMA decay for input features' second momment.
      min_norm: The minimum norm to use to avoid dividing by zero.
      cov_shrinkage: Covariance shrinkage, in [0, 1), to ensure positive
        definiteness.
      weight_decay: Weight decay factor or schedule.
      cg_ridge: Ridge regularizer for conjugate gradient.
      cg_maxiter: Number of conjugate gradient iterations.
      approx_metric: Approximate the metric with the incomplete Cholesky
        factorization, instead of using conjugate gradient to solve the linear
        system.
      is_neuron: See type definition.
      is_weight: See type definition.
      is_bias: See type definition.
      accumulator_dtype: EMA accumulator dtype.

    Returns:
      `GradientTransformationExtraArgs` with an `update_fn` that expects the
      `xs_neurons` kwarg to be a list of batched input arrays to each respective
      neuron, in leaf traversal order as specified by `is_neuron`.

    References:
      D. Munoz, `Simple Linear Neuron Boosting
      <https://arxiv.org/abs/2502.01131>`_, 2025
    """
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    return combine.chain(
        optax.ema(b_g, debias=True, accumulator_dtype=accumulator_dtype),
        scale_by_lnb(
            b_mu,
            b_nu,
            min_norm,
            cov_shrinkage,
            cg_ridge=cg_ridge,
            cg_maxiter=cg_maxiter,
            approx_metric=approx_metric,
            is_neuron=is_neuron,
            is_weight=is_weight,
            is_bias=is_bias,
            accumulator_dtype=accumulator_dtype,
        ),
        optax.add_decayed_weights(weight_decay),
    )
