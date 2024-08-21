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


.. autoclass:: Schedule

Constant schedule
~~~~~~~~~~~~~~~~~
.. autofunction:: constant_schedule

Cosine decay schedule
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cosine_decay_schedule
.. autofunction:: cosine_onecycle_schedule

Exponential decay schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: exponential_decay

Join schedules
~~~~~~~~~~~~~~
.. autofunction:: join_schedules

Inject hyperparameters
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: inject_hyperparams
.. autoclass:: InjectHyperparamsState

Linear schedules
~~~~~~~~~~~~~~~~
.. autofunction:: linear_onecycle_schedule
.. autofunction:: linear_schedule

Piecewise schedules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: piecewise_constant_schedule
.. autofunction:: piecewise_interpolate_schedule

Polynomial schedules
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: polynomial_schedule

Reduce on plateau
~~~~~~~~~~~~~~~~~
.. autofunction:: optax.contrib.reduce_on_plateau

Schedules with warm-up
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: warmup_cosine_decay_schedule
.. autofunction:: warmup_exponential_decay_schedule

Warm restarts
~~~~~~~~~~~~~
.. autofunction:: sgdr_schedule
