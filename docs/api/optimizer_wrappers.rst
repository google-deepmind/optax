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
    maybe_update
    MaybeUpdateState
    MultiSteps
    MultiStepsState
    ShouldSkipUpdateFunction
    skip_large_updates
    skip_not_finite


Apply if finite
~~~~~~~~~~~~~~~~~
.. autofunction::  apply_if_finite
.. autoclass::  ApplyIfFiniteState
   :members:

Flatten
~~~~~~~~
.. autofunction:: flatten

Lookahead
~~~~~~~~~~~~~~~~~
.. autofunction::  lookahead
.. autoclass::  LookaheadParams
   :members:
.. autoclass::  LookaheadState
   :members:

Masked update
~~~~~~~~~~~~~~
.. autofunction::  masked
.. autoclass::  MaskedState
   :members:

Maybe update
~~~~~~~~~~~~~~
.. autofunction:: maybe_update
.. autoclass:: MaybeUpdateState
   :members:

Multi-step update
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiSteps
   :members:
.. autoclass:: MultiStepsState
   :members:
