🔧 Contrib
===============

Experimental features and algorithms that don't meet the
:ref:`inclusion_criteria`.

.. currentmodule:: optax.contrib

.. autosummary::
    cocob
    COCOBState
    dadapt_adamw
    DAdaptAdamWState
    differentially_private_aggregate
    DifferentiallyPrivateAggregateState
    dpsgd
    mechanize
    MechanicState
    prodigy
    ProdigyState
    sam
    SAMState
    split_real_and_imaginary
    SplitRealAndImaginaryState

Complex-valued Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: split_real_and_imaginary
.. autoclass:: SplitRealAndImaginaryState

Continuous coin betting
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cocob
.. autoclass:: COCOBState

D-adaptation
~~~~~~~~~~~~
.. autofunction:: dadapt_adamw
.. autoclass:: DAdaptAdamWState

Differentially Private Aggregate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: differentially_private_aggregate
.. autoclass:: DifferentiallyPrivateAggregateState
.. autofunction:: dpsgd

Mechanize
~~~~~~~~~
.. autofunction:: mechanize
.. autoclass:: MechanicState

Prodigy
~~~~~~~
.. autofunction:: prodigy
.. autoclass:: ProdigyState

Sharpness aware minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sam
.. autoclass:: SAMState
