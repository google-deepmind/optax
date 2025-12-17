🔧 Contrib
===============

Algorithms or wrappers that don't meet (yet) the :ref:`inclusion_criteria` or
are not supported by the main library.

.. currentmodule:: optax.contrib

.. autosummary::
    acprop
    ademamix
    adopt
    ano
    simplified_ademamix
    cocob
    COCOBState
    dadapt_adamw
    DAdaptAdamWState
    differentially_private_aggregate
    DifferentiallyPrivateAggregateState
    dog
    DoGState
    dowg
    DoWGState
    dpsgd
    mechanize
    MechanicState
    momo
    MomoState
    momo_adam
    MomoAdamState
    muon
    MuonState
    prodigy
    ProdigyState
    sam
    SAMState
    schedule_free
    schedule_free_adamw
    schedule_free_eval_params
    schedule_free_sgd
    ScheduleFreeState
    sophia
    SophiaState
    split_real_and_imaginary
    SplitRealAndImaginaryState

AdEMAMix
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ademamix
.. autofunction:: scale_by_ademamix
.. autoclass:: ScaleByAdemamixState

Simplified AdEMAMix
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: simplified_ademamix
.. autofunction:: scale_by_simplified_ademamix
.. autoclass:: ScaleBySimplifiedAdEMAMixState

ADOPT
~~~~~
.. autofunction:: adopt
.. autofunction:: scale_by_adopt

ANO
~~~~
.. autofunction:: ano
.. autofunction:: scale_by_ano

Asynchronous-centering-Prop
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: acprop
.. autofunction:: scale_by_acprop

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

Distance over Gradients
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: dog
.. autoclass:: DoGState
.. autofunction:: dowg
.. autoclass:: DoWGState

Mechanize
~~~~~~~~~
.. autofunction:: mechanize
.. autoclass:: MechanicState

Momo
~~~~
.. autofunction:: momo
.. autoclass:: MomoState
.. autofunction:: momo_adam
.. autoclass:: MomoAdamState

Muon
~~~~
.. autofunction:: muon
.. autofunction:: scale_by_muon
.. autoclass:: MuonState

Prodigy
~~~~~~~
.. autofunction:: prodigy
.. autoclass:: ProdigyState

Schedule-Free
~~~~~~~~~~~~~
.. autofunction:: schedule_free
.. autofunction:: schedule_free_adamw
.. autofunction:: schedule_free_eval_params
.. autofunction:: schedule_free_sgd
.. autoclass:: ScheduleFreeState

Sharpness aware minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sam
.. autoclass:: SAMState

Sophia
~~~~~~
.. autofunction:: hutchinson_estimator_diag_hessian
.. autoclass:: HutchinsonState
.. autofunction:: sophia
.. autoclass:: SophiaState
