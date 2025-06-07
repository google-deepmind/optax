Projections
===========

.. currentmodule:: optax.projections

Projections can be used to perform constrained optimization.
The Euclidean projection onto a set :math:`\mathcal{C}` is:

.. math::

    \text{proj}_{\mathcal{C}}(u) :=
    \underset{v}{\text{argmin}} ~ \|u - v\|^2_2 \textrm{ subject to } v \in \mathcal{C}.

For instance, here is an example how we can project parameters to the non-negative orthant::

    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> num_weights = 2
    >>> xs = jnp.array([[-1.8, 2.2], [-2.0, 1.2]])
    >>> ys = jnp.array([0.5, 0.8])
    >>> optimizer = optax.adam(learning_rate=1e-3)
    >>> params = {'w': jnp.zeros(num_weights)}
    >>> opt_state = optimizer.init(params)
    >>> loss = lambda params, x, y: jnp.mean((params['w'].dot(x) - y) ** 2)
    >>> grads = jax.grad(loss)(params, xs, ys)
    >>> updates, opt_state = optimizer.update(grads, opt_state)
    >>> params = optax.apply_updates(params, updates)
    >>> params = optax.projections.projection_non_negative(params)

Available projections
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    projection_box
    projection_hypercube
    projection_l1_ball
    projection_l1_sphere
    projection_l2_ball
    projection_l2_sphere
    projection_linf_ball
    projection_non_negative
    projection_simplex

Projection onto a box
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_box

Projection onto a hypercube
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_hypercube

Projection onto the L1 ball
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_l1_ball

Projection onto the L1 sphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_l1_sphere

Projection onto the L2 ball
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_l2_ball

Projection onto the L2 sphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_l2_sphere

Projection onto the L-infinity ball
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_linf_ball

Projection onto the non-negative orthant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_non_negative

Projection onto a simplex
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: projection_simplex
