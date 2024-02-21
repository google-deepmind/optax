Utilities
=========

General
-------

.. currentmodule:: optax

.. autosummary::
    scale_gradient
    value_and_grad_from_state

Scale gradient
~~~~~~~~~~~~~~
.. autofunction:: scale_gradient

Value and grad from state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: value_and_grad_from_state


Numerical Stability
-------------------

.. currentmodule:: optax

.. autosummary::
    safe_int32_increment
    safe_norm
    safe_root_mean_squares

Safe int32 increment
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: safe_int32_increment

Safe norm
~~~~~~~~~
.. autofunction:: safe_norm

Safe root mean squares
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: safe_root_mean_squares



Second Order Optimization
-------------------------

.. currentmodule:: optax.second_order

.. autosummary::
    hvp_call
    make_gnvp_fn
    make_hvp_fn

Compute Hessian vector product (hvp) directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction::  hvp_call

Instantiate Gauss-Newton vector product (gnvp) function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: make_gnvp_fn

Instantiate Hessian vector product (hvp) function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: make_hvp_fn



Tree
----

.. currentmodule:: optax.tree_utils

.. autosummary::
    tree_add
    tree_add_scalar_mul
    tree_div
    tree_vdot
    tree_l2_norm
    tree_map_params
    tree_mul
    tree_ones_like
    tree_scalar_mul
    tree_sub
    tree_sum
    tree_zeros_like

Tree add
~~~~~~~~
.. autofunction:: tree_add

Tree add and scalar multiply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_add_scalar_mul

Tree divide
~~~~~~~~~~~
.. autofunction:: tree_div

Tree inner product
~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_vdot

Tree l2 norm
~~~~~~~~~~~~
.. autofunction:: tree_l2_norm

Tree map parameters
~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_map_params

Tree multiply
~~~~~~~~~~~~~
.. autofunction:: tree_mul

Tree ones like
~~~~~~~~~~~~~~
.. autofunction:: tree_ones_like

Tree scalar multiply
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_scalar_mul

Tree subtract
~~~~~~~~~~~~~
.. autofunction:: tree_sub

Tree sum
~~~~~~~~
.. autofunction:: tree_sum

Tree zeros like
~~~~~~~~~~~~~~~
.. autofunction:: tree_zeros_like
