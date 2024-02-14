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


Privacy-Sensitive Optax Methods
-------------------------------

.. currentmodule:: optax

.. autosummary::
    DifferentiallyPrivateAggregateState
    differentially_private_aggregate


Differentially Private Aggregate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: differentially_private_aggregate
.. autoclass:: DifferentiallyPrivateAggregateState
   :members:


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


Tree
----

.. currentmodule:: optax.tree_utils

.. autosummary::
    tree_map_params
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

Tree map params
~~~~~~~~~~~~~~~
.. autofunction:: tree_map_params


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
