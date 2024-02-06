Transformations
=====================

.. currentmodule:: optax

.. autosummary::
    adaptive_grad_clip
    AdaptiveGradClipState
    add_decayed_weights
    AddDecayedWeightsState
    add_noise
    AddNoiseState
    apply_every
    ApplyEvery
    bias_correction
    centralize
    clip
    clip_by_block_rms
    ClipState
    clip_by_global_norm
    ClipByGlobalNormState
    ema
    EmaState
    EmptyState
    global_norm
    GradientTransformation
    GradientTransformationExtraArgs
    identity
    keep_params_nonnegative
    NonNegativeParamsState
    OptState
    Params
    per_example_global_norm_clip
    per_example_layer_norm_clip
    scale
    ScaleState
    scale_by_adadelta
    ScaleByAdaDeltaState
    scale_by_adam
    scale_by_adamax
    ScaleByAdamState
    scale_by_amsgrad
    ScaleByAmsgradState
    scale_by_belief
    ScaleByBeliefState
    scale_by_factored_rms
    FactoredState
    scale_by_learning_rate
    scale_by_lion
    ScaleByLionState
    scale_by_novograd
    ScaleByNovogradState
    scale_by_optimistic_gradient
    scale_by_param_block_norm
    scale_by_param_block_rms
    scale_by_radam
    scale_by_rms
    ScaleByRmsState
    scale_by_rprop
    ScaleByRpropState
    scale_by_rss
    ScaleByRssState
    scale_by_schedule
    ScaleByScheduleState
    scale_by_sm3
    ScaleBySM3State
    scale_by_stddev
    ScaleByRStdDevState
    scale_by_trust_ratio
    ScaleByTrustRatioState
    scale_by_yogi
    set_to_zero
    stateless
    stateless_with_tree_map
    trace
    TraceState
    TransformInitFn
    TransformUpdateFn
    update_infinity_moment
    update_moment
    update_moment_per_elem_norm
    Updates
    with_extra_args_support
    zero_nans
    ZeroNansState


Types
~~~~~

.. autoclass:: GradientTransformation
    :members:

.. autoclass:: GradientTransformationExtraArgs
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


Transformations and states
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_grad_clip
.. autoclass:: AdaptiveGradClipState
  :members:

.. autofunction:: add_decayed_weights
.. autoclass:: AddDecayedWeightsState
    :members:

.. autofunction:: add_noise
.. autoclass:: AddNoiseState
    :members:

.. autofunction:: apply_every
.. autoclass:: ApplyEvery
    :members:

.. autofunction:: bias_correction

.. autofunction:: centralize

.. autofunction:: clip
.. autofunction:: clip_by_block_rms
.. autoclass:: ClipState
    :members:

.. autofunction:: clip_by_global_norm
.. autoclass:: ClipByGlobalNormState
    :members:

.. autofunction:: ema
.. autoclass:: EmaState
    :members:

.. autoclass:: EmptyState
    :members:

.. autofunction:: global_norm

.. autofunction:: identity

.. autofunction:: keep_params_nonnegative
.. autoclass:: NonNegativeParamsState
    :members:

.. autofunction:: per_example_global_norm_clip
.. autofunction:: per_example_layer_norm_clip

.. autofunction:: scale
.. autoclass:: ScaleState
    :members:

.. autofunction:: scale_by_adadelta
.. autoclass:: ScaleByAdaDeltaState
    :members:

.. autofunction:: scale_by_adam
.. autofunction:: scale_by_adamax
.. autoclass:: ScaleByAdamState
    :members:

.. autofunction:: scale_by_amsgrad
.. autoclass:: ScaleByAmsgradState
    :members:

.. autofunction:: scale_by_belief
.. autoclass:: ScaleByBeliefState
    :members:

.. autofunction:: scale_by_factored_rms
.. autoclass:: FactoredState
    :members:

.. autofunction:: scale_by_learning_rate

.. autofunction:: scale_by_lion
.. autoclass:: ScaleByLionState
    :members:

.. autofunction:: scale_by_novograd
.. autoclass:: ScaleByNovogradState
    :members:

.. autofunction:: scale_by_optimistic_gradient

.. autofunction:: scale_by_param_block_norm

.. autofunction:: scale_by_param_block_rms

.. autofunction:: scale_by_radam

.. autofunction:: scale_by_rms
.. autoclass:: ScaleByRmsState
    :members:

.. autofunction:: scale_by_rprop
.. autoclass:: ScaleByRpropState
    :members:

.. autofunction:: scale_by_rss
.. autoclass:: ScaleByRssState
    :members:

.. autofunction:: scale_by_schedule
.. autoclass:: ScaleByScheduleState
    :members:

.. autofunction:: scale_by_sm3
.. autoclass:: ScaleBySM3State
    :members:

.. autofunction:: scale_by_stddev
.. autoclass:: ScaleByRStdDevState
    :members:

.. autofunction:: scale_by_trust_ratio
.. autoclass:: ScaleByTrustRatioState
    :members:

.. autofunction:: scale_by_yogi

.. autofunction:: set_to_zero

.. autofunction:: stateless
.. autofunction:: stateless_with_tree_map

.. autofunction:: trace
.. autoclass:: TraceState
    :members:

.. autofunction:: update_infinity_moment
.. autofunction:: update_moment
.. autofunction:: update_moment_per_elem_norm

.. autofunction:: with_extra_args_support

.. autofunction:: zero_nans
.. autoclass:: ZeroNansState
    :members:
