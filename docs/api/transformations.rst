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
    conditionally_mask
    conditionally_transform
    ConditionallyMaskState
    ConditionallyTransformState
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
    init_empty_state
    keep_params_nonnegative
    measure_with_ema
    monitor
    MonitorState
    NonNegativeParamsState
    normalize_by_update_norm
    OptState
    Params
    per_example_global_norm_clip
    per_example_layer_norm_clip
    scale
    ScaleState
    scale_by_adadelta
    ScaleByAdaDeltaState
    scale_by_adan
    ScaleByAdanState
    scale_by_adam
    scale_by_adamax
    ScaleByAdamState
    scale_by_amsgrad
    ScaleByAmsgradState
    scale_by_backtracking_linesearch
    ScaleByBacktrackingLinesearchState
    scale_by_belief
    ScaleByBeliefState
    scale_by_factored_rms
    FactoredState
    scale_by_lbfgs
    ScaleByLBFGSState
    scale_by_learning_rate
    scale_by_lion
    ScaleByLionState
    scale_by_novograd
    ScaleByNovogradState
    scale_by_optimistic_gradient
    scale_by_param_block_norm
    scale_by_param_block_rms
    scale_by_polyak
    scale_by_radam
    scale_by_rms
    ScaleByRmsState
    scale_by_rprop
    ScaleByRpropState
    scale_by_rss
    ScaleByRssState
    scale_by_schedule
    ScaleByScheduleState
    scale_by_sign
    scale_by_sm3
    ScaleBySM3State
    scale_by_stddev
    ScaleByRStdDevState
    scale_by_trust_ratio
    ScaleByTrustRatioState
    scale_by_yogi
    scale_by_zoom_linesearch
    ScaleByZoomLinesearchState
    set_to_zero
    snapshot
    SnapshotState
    stateless
    stateless_with_tree_map
    trace
    TraceState
    TransformInitFn
    TransformUpdateFn
    TransformUpdateExtraArgsFn
    update_infinity_moment
    update_moment
    update_moment_per_elem_norm
    Updates
    with_extra_args_support
    zero_nans
    ZeroNansState
    ZoomLinesearchInfo


Types
~~~~~

.. autoclass:: GradientTransformation

.. autoclass:: GradientTransformationExtraArgs

.. autoclass:: TransformInitFn

.. autoclass:: TransformUpdateFn

.. autoclass:: TransformUpdateExtraArgsFn

.. autoclass:: OptState

.. autoclass:: Params

.. autoclass:: Updates


Transformations and states
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_grad_clip
.. autoclass:: AdaptiveGradClipState

.. autofunction:: add_decayed_weights
.. autoclass:: AddDecayedWeightsState

.. autofunction:: add_noise
.. autoclass:: AddNoiseState

.. autofunction:: apply_every
.. autoclass:: ApplyEvery

.. autofunction:: bias_correction

.. autofunction:: centralize

.. autofunction:: conditionally_mask
.. autoclass:: ConditionallyMaskState

.. autofunction:: conditionally_transform
.. autoclass:: ConditionallyTransformState

.. autofunction:: clip
.. autofunction:: clip_by_block_rms
.. autoclass:: ClipState

.. autofunction:: clip_by_global_norm
.. autoclass:: ClipByGlobalNormState

.. autofunction:: ema
.. autoclass:: EmaState

.. autoclass:: EmptyState

.. autofunction:: global_norm

.. autofunction:: identity

.. autofunction:: keep_params_nonnegative
.. autoclass:: NonNegativeParamsState

.. autofunction:: masked

.. autofunction:: normalize_by_update_norm

.. autofunction:: per_example_global_norm_clip
.. autofunction:: per_example_layer_norm_clip

.. autofunction:: scale
.. autoclass:: ScaleState

.. autofunction:: scale_by_adadelta
.. autoclass:: ScaleByAdaDeltaState

.. autofunction:: scale_by_adan
.. autoclass:: ScaleByAdanState

.. autofunction:: scale_by_adam
.. autofunction:: scale_by_adamax
.. autoclass:: ScaleByAdamState

.. autofunction:: scale_by_amsgrad
.. autoclass:: ScaleByAmsgradState

.. autofunction:: scale_by_backtracking_linesearch
.. autoclass:: ScaleByBacktrackingLinesearchState

.. autofunction:: scale_by_belief
.. autoclass:: ScaleByBeliefState

.. autofunction:: scale_by_factored_rms
.. autoclass:: FactoredState

.. autofunction:: scale_by_lbfgs
.. autoclass:: ScaleByLBFGSState

.. autofunction:: scale_by_learning_rate

.. autofunction:: scale_by_lion
.. autoclass:: ScaleByLionState

.. autofunction:: scale_by_novograd
.. autoclass:: ScaleByNovogradState

.. autofunction:: scale_by_optimistic_gradient

.. autofunction:: scale_by_param_block_norm

.. autofunction:: scale_by_param_block_rms

.. autofunction:: scale_by_radam

.. autofunction:: scale_by_polyak

.. autofunction:: scale_by_rms
.. autoclass:: ScaleByRmsState

.. autofunction:: scale_by_rprop
.. autoclass:: ScaleByRpropState

.. autofunction:: scale_by_rss
.. autoclass:: ScaleByRssState

.. autofunction:: scale_by_schedule
.. autoclass:: ScaleByScheduleState

.. autofunction:: scale_by_sign

.. autofunction:: scale_by_sm3
.. autoclass:: ScaleBySM3State

.. autofunction:: scale_by_stddev
.. autoclass:: ScaleByRStdDevState

.. autofunction:: scale_by_trust_ratio
.. autoclass:: ScaleByTrustRatioState

.. autofunction:: scale_by_yogi

.. autofunction:: scale_by_zoom_linesearch
.. autoclass:: ScaleByZoomLinesearchState

.. autofunction:: set_to_zero

.. autofunction:: stateless
.. autofunction:: stateless_with_tree_map

.. autofunction:: snapshot
.. autoclass:: SnapshotState

.. autofunction:: trace
.. autoclass:: TraceState

.. autofunction:: update_infinity_moment
.. autofunction:: update_moment
.. autofunction:: update_moment_per_elem_norm

.. autofunction:: with_extra_args_support

.. autofunction:: zero_nans
.. autoclass:: ZeroNansState

.. autoclass:: ZoomLinesearchInfo


Freezing
~~~~~~~~

.. autofunction:: freeze
.. autofunction:: selective_transform
