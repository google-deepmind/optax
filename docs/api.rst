Common Optimizers
===================

.. currentmodule:: optax

.. autosummary::

    adabelief
    adagrad
    adam
    adamw
    fromage
    lamb
    noisy_sgd
    dpsgd
    radam
    rmsprop
    sgd
    yogi


AdaBelief
~~~~~~~~~

.. autofunction:: adabelief

AdaGrad
~~~~~~~

.. autofunction:: adagrad

Adam
~~~~

.. autofunction:: adam

AdamW
~~~~~

.. autofunction:: adamw

Fromage
~~~~~~~

.. autofunction:: fromage

Lamb
~~~~

.. autofunction:: lamb

Noisy SGD
~~~~~~~~~

.. autofunction:: noisy_sgd

Differentially Private SGD
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dpsgd

RAdam
~~~~~

.. autofunction:: radam

RMSProp
~~~~~~~

.. autofunction:: rmsprop

SGD
~~~

.. autofunction:: sgd

Yogi
~~~~

.. autofunction:: yogi


Optax Transformations
=====================


Gradient Transforms
-------------------

.. currentmodule:: optax

.. autosummary::

    adaptive_grad_clip
    add_decayed_weights
    add_noise
    AddDecayedWeightsState
    additive_weight_decay
    AdditiveWeightDecayState
    AddNoiseState
    apply_every
    ApplyEvery
    centralize
    clip
    clip_by_global_norm
    ClipByGlobalNormState
    ClipState
    global_norm
    GradientTransformation
    identity
    keep_params_nonnegative
    NonNegativeParamsState
    OptState
    Params
    scale
    scale_by_adam
    scale_by_belief
    scale_by_radam
    scale_by_rms
    scale_by_rss
    scale_by_schedule
    scale_by_stddev
    scale_by_trust_ratio
    scale_by_yogi
    ScaleByAdamState
    ScaleByFromageState
    ScaleByRmsState
    ScaleByRssState
    ScaleByRStdDevState
    ScaleByScheduleState
    ScaleByTrustRatioState
    ScaleState
    trace
    TraceState
    TransformInitFn
    TransformUpdateFn
    Updates
    zero_nans
    ZeroNansState


Optax Types
~~~~~~~~~~~~~~

.. autoclass:: GradientTransformation
    :members:

.. autoclass:: OptState
    :members:

.. autoclass:: Params
    :members:

.. autoclass:: Updates
    :members:

.. autoclass:: TransformInitFn
    :members:

.. autoclass:: TransformUpdateFn
    :members:


Optax Transforms and States
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_grad_clip
.. autoclass:: AdaptiveGradClipState
  :members:

.. autofunction:: add_decayed_weights
.. autofunction:: add_noise
.. autoclass:: AddDecayedWeightsState
    :members:

.. autofunction:: additive_weight_decay
.. autoclass:: AdditiveWeightDecayState
    :members:

.. autoclass:: AddNoiseState
    :members:

.. autofunction:: apply_every
.. autoclass:: ApplyEvery
    :members:

.. autofunction:: centralize
.. autofunction:: clip
.. autofunction:: clip_by_global_norm
.. autoclass:: ClipByGlobalNormState
    :members:

.. autoclass:: ClipState
    :members:

.. autofunction:: global_norm
.. autofunction:: identity
.. autofunction:: keep_params_nonnegative
.. autoclass:: NonNegativeParamsState
    :members:

.. autofunction:: scale
.. autofunction:: scale_by_adam
.. autofunction:: scale_by_belief
.. autofunction:: scale_by_radam
.. autofunction:: scale_by_rms
.. autofunction:: scale_by_rss
.. autofunction:: scale_by_schedule
.. autofunction:: scale_by_stddev
.. autofunction:: scale_by_trust_ratio
.. autofunction:: scale_by_yogi
.. autoclass:: ScaleByAdamState
    :members:

.. autoclass:: ScaleByFromageState
    :members:

.. autoclass:: ScaleByRmsState
    :members:

.. autoclass:: ScaleByRssState
    :members:

.. autoclass:: ScaleByRStdDevState
    :members:

.. autoclass:: ScaleByScheduleState
    :members:

.. autoclass:: ScaleByTrustRatioState
    :members:

.. autoclass:: ScaleState
    :members:

.. autofunction:: trace
.. autoclass:: TraceState
    :members:

.. autofunction:: zero_nans
.. autoclass:: ZeroNansState
    :members:



Apply Updates
=============

.. autosummary::
    apply_updates
    incremental_update
    periodic_update

apply_updates
~~~~~~~~~~~~~~~~~

.. autofunction:: apply_updates

incremental_update
~~~~~~~~~~~~~~~~~~

.. autofunction:: incremental_update

periodic_update
~~~~~~~~~~~~~~~

.. autofunction:: periodic_update



Combining Optimizers
=====================

.. currentmodule:: optax

.. autosummary::

    chain

chain
~~~~~

.. autofunction:: chain




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


apply_if_finite
~~~~~~~~~~~~~~~~~

.. autofunction::  apply_if_finite


ApplyIfFiniteState
~~~~~~~~~~~~~~~~~~~

.. autoclass::  ApplyIfFiniteState
   :members:


flatten
~~~~~~~~

.. autofunction:: flatten


lookahead
~~~~~~~~~~~~~~~~~

.. autofunction::  lookahead

LookaheadParams
~~~~~~~~~~~~~~~~~

.. autoclass::  LookaheadParams
   :members:

LookaheadState
~~~~~~~~~~~~~~~~~

.. autoclass::  LookaheadState
   :members:


Masked Update
~~~~~~~~~~~~~~

.. autofunction::  masked

.. autoclass::  MaskedState
   :members:



Maybe Update
~~~~~~~~~~~~~~

.. autofunction::maybe_update
.. autoclass::MaybeUpdateState
   :members:


Multi-step Update
~~~~~~~~~~~~~~~~~~~~

.. autoclass::MultiSteps
   :members:

.. autoclass::MultiStepsState
   :members:


Common Losses
===============

.. currentmodule:: optax

.. autosummary::

    cosine_distance
    cosine_similarity
    huber_loss
    l2_loss
    sigmoid_binary_cross_entropy
    smooth_labels
    softmax_cross_entropy


losses
~~~~~~~

.. autofunction:: cosine_distance
.. autofunction:: cosine_similarity
.. autofunction:: huber_loss
.. autofunction:: l2_loss
.. autofunction:: sigmoid_binary_cross_entropy
.. autofunction:: smooth_labels
.. autofunction:: softmax_cross_entropy



Linear Algebra Operators
========================

.. currentmodule:: optax

.. autosummary::

    matrix_inverse_pth_root
    power_iteration


matrix_inverse_pth_root
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: matrix_inverse_pth_root


power_iteration
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: power_iteration


Optimizer Schedules
=====================

.. currentmodule:: optax

.. autosummary::

    constant_schedule
    cosine_decay_schedule
    cosine_onecycle_schedule
    exponential_decay
    linear_onecycle_schedule
    piecewise_constant_schedule
    piecewise_interpolate_schedule
    polynomial_schedule
    Schedule
    InjectHyperparamsState
    inject_hyperparams

schedules
~~~~~~~~~

.. autofunction:: constant_schedule
.. autofunction:: cosine_decay_schedule
.. autofunction:: cosine_onecycle_schedule
.. autofunction:: exponential_decay
.. autofunction:: linear_onecycle_schedule
.. autofunction:: piecewise_constant_schedule
.. autofunction:: piecewise_interpolate_schedule
.. autofunction:: polynomial_schedule
.. autofunction:: inject_hyperparams

.. autoclass:: Schedule
   :members:

.. autoclass:: InjectHyperparamsState
   :members:



Second Order Optimization Utilities
=====================================

.. currentmodule:: optax

.. autosummary::

    fisher_diag
    hessian_diag
    hvp

fisher_diag
~~~~~~~~~~~

.. autofunction:: fisher_diag

hessian_diag
~~~~~~~~~~~~~~~~~

.. autofunction:: hessian_diag

hvp
~~~~~~~~~~~

.. autofunction:: hvp






Control Variates
================

.. currentmodule:: optax

.. autosummary::

    control_delta_method
    control_variates_jacobians
    moving_avg_baseline

control_delta_method
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: control_delta_method

control_variates_jacobians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: control_variates_jacobians

moving_avg_baseline
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: moving_avg_baseline




Stochastic Gradient Estimators
==============================

.. currentmodule:: optax

.. autosummary::

    measure_valued_jacobians
    pathwise_jacobians
    score_function_jacobians

measure_valued_jacobians
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: measure_valued_jacobians

pathwise_jacobians
~~~~~~~~~~~~~~~~~~

.. autofunction:: pathwise_jacobians

score_function_jacobians
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: score_function_jacobians



Privacy-Sensitive Optax Methods
==================================

.. currentmodule:: optax

.. autosummary::

    DifferentiallyPrivateAggregateState
    differentially_private_aggregate


differentially_private_aggregate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: differentially_private_aggregate

.. autoclass:: DifferentiallyPrivateAggregateState
   :members:
