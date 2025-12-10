🧪 Experimental
===============

Experimental features subject to changes before being graduated into `optax`.

.. currentmodule:: optax.experimental

.. autosummary::
  microbatching.microbatch
  microbatching.micro_vmap
  microbatching.micro_grad
  microbatching.AccumulationType
  microbatching.Accumulator

.. currentmodule:: optax.experimental.microbatching

Microbatching
~~~~~~~~~~~~~
.. autoclass:: AccumulationType
   :members:
.. autofunction:: microbatch
.. autofunction:: micro_vmap
.. autofunction:: micro_grad
.. autofunction:: reshape_batch_axis
