Optimizer Schedules
=====================

.. currentmodule:: optax

.. autosummary::

    constant_schedule
    cosine_decay_schedule
    cosine_onecycle_schedule
    exponential_decay
    join_schedules
    linear_onecycle_schedule
    linear_schedule
    piecewise_constant_schedule
    piecewise_interpolate_schedule
    polynomial_schedule
    sgdr_schedule
    warmup_cosine_decay_schedule
    warmup_exponential_decay_schedule
    Schedule
    InjectHyperparamsState
    inject_hyperparams

Schedules
~~~~~~~~~

.. autofunction:: constant_schedule
.. autofunction:: cosine_decay_schedule
.. autofunction:: cosine_onecycle_schedule
.. autofunction:: exponential_decay
.. autofunction:: join_schedules
.. autofunction:: linear_onecycle_schedule
.. autofunction:: linear_schedule
.. autofunction:: piecewise_constant_schedule
.. autofunction:: piecewise_interpolate_schedule
.. autofunction:: polynomial_schedule
.. autofunction:: optax.contrib.reduce_on_plateau
.. autofunction:: sgdr_schedule
.. autofunction:: warmup_cosine_decay_schedule
.. autofunction:: warmup_exponential_decay_schedule
.. autofunction:: inject_hyperparams

.. autoclass:: Schedule
   :members:

.. autoclass:: InjectHyperparamsState
   :members:



