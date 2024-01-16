🔧 Contrib
===============

.. currentmodule:: optax.contrib

.. autosummary::
    cocob
    COCOBState
    dadapt_adamw
    DAdaptAdamWState
    mechanize
    MechanicState
    polyak_step_sgd
    PolyakStepSGDState
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
    :members:

Continuous coin betting
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cocob
.. autoclass:: COCOBState
    :members:

D-adaptation
~~~~~~~~~~~~
.. autofunction:: dadapt_adamw
.. autoclass:: DAdaptAdamWState
    :members:

Mechanize
~~~~~~~~~
.. autofunction:: mechanize
.. autoclass:: MechanicState
    :members:

Polyak step-size
~~~~~~~~~~~~~~~~
.. autofunction:: polyak_step_sgd
.. autoclass:: PolyakStepSGDState
    :members:


Prodigy
~~~~~~~
.. autofunction:: prodigy
.. autoclass:: ProdigyState
    :members:

Sharpness aware minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sam
.. autoclass:: SAMState
    :members:
