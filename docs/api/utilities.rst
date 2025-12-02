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
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: value_and_grad_from_state


Numerical Stability
-------------------

.. currentmodule:: optax

.. autosummary::
    safe_increment
    safe_norm
    safe_root_mean_squares

Safe increment
~~~~~~~~~~~~~~
.. autofunction:: safe_increment

Safe norm
~~~~~~~~~
.. autofunction:: safe_norm

Safe root mean squares
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: safe_root_mean_squares


Linear Algebra Operators
------------------------

.. currentmodule:: optax

.. autosummary::
    matrix_inverse_pth_root
    power_iteration
    nnls

Matrix inverse pth root
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: matrix_inverse_pth_root

Power iteration
~~~~~~~~~~~~~~~
.. autofunction:: power_iteration

Non-negative least squares
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nnls


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


State utilities
-------------------

.. currentmodule:: optax.tree_utils

.. autosummary::
    ParamsShapedState
    reshape_params_shaped_state
    shape_state_like_params
    tree_map_params
    tree_get
    tree_get_all_with_path
    tree_set

ParamsShapedState
~~~~~~~~~~~~~~~~~
.. autoclass:: ParamsShapedState

Reshape params-shaped state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: reshape_params_shaped_state

Shape state like params
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: shape_state_like_params

Fetch single value that match a given key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_get

Fetch all values that match a given key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_get_all_with_path

Tree map parameters
~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_map_params

Set values in a tree
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_set


Tree
----

.. currentmodule:: optax.tree_utils

.. autosummary::
    NamedTupleKey
    tree_add
    tree_add_scale
    tree_allclose
    tree_batch_shape
    tree_cast
    tree_cast_like
    tree_clip
    tree_conj
    tree_div
    tree_dtype
    tree_full_like
    tree_norm
    tree_max
    tree_min
    tree_mul
    tree_ones_like
    tree_random_like
    tree_real
    tree_split_key_like
    tree_scale
    tree_set
    tree_size
    tree_sub
    tree_sum
    tree_vdot
    tree_where
    tree_zeros_like

NamedTupleKey
~~~~~~~~~~~~~
.. autoclass:: NamedTupleKey

Tree add
~~~~~~~~
.. autofunction:: tree_add

Tree add and scalar multiply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_add_scale

Tree all close
~~~~~~~~~~~~~~
.. autofunction:: tree_allclose

Tree batch reshaping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_batch_shape

Tree cast
~~~~~~~~~
.. autofunction:: tree_cast

Tree cast like
~~~~~~~~~~~~~~
.. autofunction:: tree_cast_like

Tree clip
~~~~~~~~~
.. autofunction:: tree_clip

Tree conjugate
~~~~~~~~~~~~~~
.. autofunction:: tree_conj

Tree data type
~~~~~~~~~~~~~~
.. autofunction:: tree_dtype

Tree full like
~~~~~~~~~~~~~~
.. autofunction:: tree_full_like

Tree divide
~~~~~~~~~~~
.. autofunction:: tree_div


Tree norm
~~~~~~~~~
.. autofunction:: tree_norm

Tree max
~~~~~~~~
.. autofunction:: tree_max

Tree min
~~~~~~~~
.. autofunction:: tree_min

Tree multiply
~~~~~~~~~~~~~
.. autofunction:: tree_mul

Tree ones like
~~~~~~~~~~~~~~
.. autofunction:: tree_ones_like

Split key according to structure of a tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_split_key_like

Tree with random values
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_random_like

Tree real part
~~~~~~~~~~~~~~
.. autofunction:: tree_real

Tree scalar multiply
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_scale

Tree size
~~~~~~~~~
.. autofunction:: tree_size

Tree subtract
~~~~~~~~~~~~~
.. autofunction:: tree_sub

Tree sum
~~~~~~~~
.. autofunction:: tree_sum

Tree inner product
~~~~~~~~~~~~~~~~~~
.. autofunction:: tree_vdot

Tree where
~~~~~~~~~~
.. autofunction:: tree_where

Tree zeros like
~~~~~~~~~~~~~~~
.. autofunction:: tree_zeros_like
