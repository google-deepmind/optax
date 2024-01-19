Utilities
=========

General
-------

.. currentmodule:: optax

.. autosummary::
    scale_gradient

Scale gradient
~~~~~~~~~~~~~~
.. autofunction:: scale_gradient


Tree
----

.. currentmodule:: optax.tree_utils

.. autosummary::
    tree_map_params
    tree_add
    tree_sub
    tree_mul
    tree_div
    tree_scalar_mul
    tree_add_scalar_mul
    tree_vdot
    tree_sum
    tree_l2_norm
    tree_zeros_like
    tree_ones_like

Tree map parameters
~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_map_params

Tree add
~~~~~~~~
.. autofunction:: tree_add

Tree subtract
~~~~~~~~~~~~~
.. autofunction:: tree_sub

Tree multiply
~~~~~~~~~~~~~
.. autofunction:: tree_mul

Tree divide
~~~~~~~~~~~
.. autofunction:: tree_div

Tree scalar multiply
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_scalar_mul

Tree add and scalar multiply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_add_scalar_mul

Tree inner product
~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_vdot

Tree sum
~~~~~~~~
.. autofunction:: tree_sum

Tree l2 norm
~~~~~~~~~~~~
.. autofunction:: tree_l2_norm

Tree zeros like
~~~~~~~~~~~~~~~
.. autofunction:: tree_zeros_like

Tree ones like
~~~~~~~~~~~~~~
.. autofunction:: tree_ones_like


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
    fisher_diag
    hessian_diag
    hvp

Fisher diagonal
~~~~~~~~~~~~~~~
.. autofunction:: fisher_diag

Hessian diagonal
~~~~~~~~~~~~~~~~
.. autofunction:: hessian_diag

Hessian vector product
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: hvp
