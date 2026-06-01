Utilities
=========

General
-------

.. currentmodule:: optax

.. autosummary::
    :toctree: generated/

    scale_gradient
    value_and_grad_from_state


Numerical Stability
-------------------

.. currentmodule:: optax

.. autosummary::
    :toctree: generated/

    safe_increment
    safe_norm
    safe_root_mean_squares


Linear Algebra Operators
------------------------

.. currentmodule:: optax

.. autosummary::
    :toctree: generated/

    matrix_inverse_pth_root
    power_iteration
    nnls


Second Order Optimization
-------------------------

.. currentmodule:: optax.second_order

.. autosummary::
    :toctree: generated/

    fisher_diag
    hessian_diag
    hvp


Tree
----

.. currentmodule:: optax.tree_utils

.. autosummary::
    :toctree: generated/

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
    tree_get
    tree_get_all_with_path
    tree_norm
    tree_map_params
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
