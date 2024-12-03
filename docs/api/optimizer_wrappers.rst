Optimizer Wrappers
====================

.. currentmodule:: optax

.. autosummary::
    apply_if_finite
    ApplyIfFiniteState
    flatten
    lookahead
    LookaheadParams
    LookaheadState
    masked
    MaskedState
    MultiSteps
    MultiStepsState
    ShouldSkipUpdateFunction
    skip_large_updates
    skip_not_finite


Apply if finite
~~~~~~~~~~~~~~~
.. autofunction::  apply_if_finite
.. autoclass::  ApplyIfFiniteState

Flatten
~~~~~~~~
.. autofunction:: flatten

Lookahead
~~~~~~~~~
.. autofunction::  lookahead
.. autoclass::  LookaheadParams
.. autoclass::  LookaheadState

Masked update
~~~~~~~~~~~~~
.. autofunction::  masked
.. autoclass::  MaskedState

Maybe update
~~~~~~~~~~~~
.. autofunction:: maybe_update
.. autoclass:: MaybeUpdateState

Multi-step update
~~~~~~~~~~~~~~~~~
.. autoclass:: MultiSteps
.. autoclass:: MultiStepsState
.. autoclass:: ShouldSkipUpdateFunction
.. autofunction:: skip_large_updates
.. autofunction:: skip_not_finite
