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


Apply if Finite
~~~~~~~~~~~~~~~~~

.. autofunction::  apply_if_finite

.. autoclass::  ApplyIfFiniteState
   :members:


flatten
~~~~~~~~

.. autofunction:: flatten


Lookahead
~~~~~~~~~~~~~~~~~

.. autofunction::  lookahead

.. autoclass::  LookaheadParams
   :members:

.. autoclass::  LookaheadState
   :members:


Masked Update
~~~~~~~~~~~~~~

.. autofunction::  masked

.. autoclass::  MaskedState
   :members:



Maybe Update
~~~~~~~~~~~~~~

.. autofunction:: maybe_update
.. autoclass:: MaybeUpdateState
   :members:


Multi-step Update
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiSteps
   :members:

.. autoclass:: MultiStepsState
   :members:


