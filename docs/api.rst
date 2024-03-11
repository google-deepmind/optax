Common Optimizers
===================

.. currentmodule:: optax

.. autosummary::

    adabelief
    adafactor
    adagrad
    adam
    adamw
    adamax
    adamaxw
    amsgrad
    eve
    fromage
    lamb
    lars
    noisy_sgd
    novograd
    optimistic_gradient_descent
    dpsgd
    radam
    rmsprop
    sgd
    sm3
    yogi


AdaBelief
~~~~~~~~~

.. autofunction:: adabelief

AdaGrad
~~~~~~~

.. autofunction:: adagrad

AdaFactor
~~~~~~~~~

.. autofunction:: adafactor

Adam
~~~~

.. autofunction:: adam

Adamax
~~~~

.. autofunction:: adamax

AdamaxW
~~~~~

.. autofunction:: adamaxw

AdamW
~~~~~

.. autofunction:: adamw

AMSGrad
~~~~~

.. autofunction:: amsgrad

Eve
~~~

.. autofunction:: eve

Fromage
~~~~~~~

.. autofunction:: fromage

Lamb
~~~~

.. autofunction:: lamb

Lars
~~~~

.. autofunction:: lars

SM3
~~~

.. autofunction:: sm3


Noisy SGD
~~~~~~~~~

.. autofunction:: noisy_sgd


Novograd
~~~~~~~~~

.. autofunction:: novograd


Optimistic GD
~~~~~~~~~~~~~

.. autofunction:: optimistic_gradient_descent


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
    bias_correction
    centralize
    clip
    clip_by_block_rms
    clip_by_global_norm
    ClipByGlobalNormState
    ClipState
    ema
    EmaState
    EmptyState
    FactoredState
    global_norm
    GradientTransformation
    identity
    keep_params_nonnegative
    NonNegativeParamsState
    OptState
    Params
    scale
    scale_by_adam
    scale_by_adamax
    scale_by_amsgrad
    scale_by_belief
    scale_by_factored_rms
    scale_by_novograd
    scale_by_optimistic_gradient
    scale_by_param_block_norm
    scale_by_param_block_rms
    scale_by_radam
    scale_by_rms
    scale_by_rss
    scale_by_schedule
    scale_by_sm3
    scale_by_stddev
    scale_by_trust_ratio
    scale_by_yogi
    ScaleByAdamState
    ScaleByAmsgradState
    ScaleByNovogradState
    ScaleByRmsState
    ScaleByRssState
    ScaleByRStdDevState
    ScaleByScheduleState
    ScaleByTrustRatioState
    ScaleBySM3State
    ScaleState
    stateless
    stateless_with_tree_map
    set_to_zero
    trace
    TraceState
    TransformInitFn
    TransformUpdateFn
    update_infinity_moment
    update_moment
    update_moment_per_elem_norm
    Updates
    zero_nans
    ZeroNansState


Optax Types
~~~~~~~~~~~~~~

.. autoclass:: GradientTransformation
    :members:

.. autoclass:: TransformInitFn
    :members:

.. autoclass:: TransformUpdateFn
    :members:

.. autoclass:: OptState
    :members:

.. autoclass:: Params
    :members:

.. autoclass:: Updates
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
.. autofunction:: clip_by_block_rms
.. autofunction:: clip_by_global_norm
.. autoclass:: ClipByGlobalNormState
    :members:

.. autoclass:: ClipState
    :members:

.. autofunction:: ema
.. autoclass:: EmaState
    :members:

.. autoclass:: EmptyState
    :members:

.. autoclass:: FactoredState
    :members:

.. autofunction:: global_norm
.. autofunction:: identity
.. autofunction:: keep_params_nonnegative
.. autoclass:: NonNegativeParamsState
    :members:

.. autofunction:: scale
.. autofunction:: scale_by_adam
.. autofunction:: scale_by_adamax
.. autofunction:: scale_by_amsgrad
.. autofunction:: scale_by_belief
.. autofunction:: scale_by_eve
.. autofunction:: scale_by_factored_rms
.. autofunction:: scale_by_novograd
.. autofunction:: scale_by_param_block_norm
.. autofunction:: scale_by_param_block_rms
.. autofunction:: scale_by_radam
.. autofunction:: scale_by_rms
.. autofunction:: scale_by_rss
.. autofunction:: scale_by_schedule
.. autofunction:: scale_by_sm3
.. autofunction:: scale_by_stddev
.. autofunction:: scale_by_trust_ratio
.. autofunction:: scale_by_yogi
.. autoclass:: ScaleByAdamState
    :members:

.. autoclass:: ScaleByAmsgradState
    :members:

.. autoclass:: ScaleByNovogradState
    :members:

.. autoclass:: ScaleByEveState
    :members:

.. autoclass:: ScaleByRmsState
    :members:

.. autoclass:: ScaleByRssState
    :members:


.. autoclass:: ScaleByRStdDevState
    :members:

.. autoclass:: ScaleByScheduleState
    :members:

.. autoclass:: ScaleBySM3State
    :members:

.. autoclass:: ScaleByTrustRatioState
    :members:

.. autoclass:: ScaleState
    :members:

.. autofunction:: set_to_zero

.. autofunction:: stateless
.. autofunction:: stateless_with_tree_map

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
    multi_transform

chain
~~~~~

.. autofunction:: chain


Multi Transform
~~~~~~~~~~~~~~~

.. autofunction:: multi_transform
.. autoclass::  MultiTransformState
   :members:


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


Common Losses
===============

.. currentmodule:: optax

.. autosummary::

    cosine_distance
    cosine_similarity
    ctc_loss
    ctc_loss_with_forward_probs
    hinge_loss
    huber_loss
    l2_loss
    log_cosh
    sigmoid_binary_cross_entropy
    smooth_labels
    softmax_cross_entropy
    softmax_cross_entropy_with_integer_labels


Losses
~~~~~~~

.. autofunction:: cosine_distance
.. autofunction:: cosine_similarity
.. autofunction:: ctc_loss
.. autofunction:: ctc_loss_with_forward_probs
.. autofunction:: hinge_loss
.. autofunction:: huber_loss
.. autofunction:: l2_loss
.. autofunction:: log_cosh
.. autofunction:: sigmoid_binary_cross_entropy
.. autofunction:: smooth_labels
.. autofunction:: softmax_cross_entropy
.. autofunction:: softmax_cross_entropy_with_integer_labels



Linear Algebra Operators
========================

.. currentmodule:: optax

.. autosummary::

    matrix_inverse_pth_root
    multi_normal
    power_iteration


multi_normal
~~~~~~~~~~~~
.. autofunction:: multi_normal


matrix_inverse_pth_root
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: matrix_inverse_pth_root


Utilities for numerical stability
=================================

.. currentmodule:: optax

.. autosummary::

    safe_int32_increment
    safe_norm
    safe_root_mean_squares


Numerics
~~~~~~~~

.. autofunction:: safe_int32_increment
.. autofunction:: safe_norm
.. autofunction:: safe_root_mean_squares


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
.. autofunction:: sgdr_schedule
.. autofunction:: warmup_cosine_decay_schedule
.. autofunction:: warmup_exponential_decay_schedule
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



General Utilities
=====================================

.. currentmodule:: optax

.. autosummary::

    multi_normal
    scale_gradient

multi_normal
~~~~~~~~~~~~

.. autofunction:: multi_normal

scale_gradient
~~~~~~~~~~~~~~~~~

.. autofunction:: scale_gradient


🚧 Experimental
===============

.. currentmodule:: optax.experimental

.. autosummary::

    split_real_and_imaginary
    SplitRealAndImaginaryState


Complex-Valued Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction::  split_real_and_imaginary

.. autoclass::  SplitRealAndImaginaryState
   :members:
