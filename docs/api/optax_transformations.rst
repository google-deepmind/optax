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
    GradientTransformationExtraArgs
    identity
    keep_params_nonnegative
    NonNegativeParamsState
    OptState
    Params
    per_example_global_norm_clip
    per_example_layer_norm_clip
    scale
    scale_by_adam
    scale_by_adamax
    scale_by_amsgrad
    scale_by_belief
    scale_by_factored_rms
    scale_by_lion
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
    ScaleByLionState
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
    tree_map_params
    TraceState
    TransformInitFn
    TransformUpdateFn
    update_infinity_moment
    update_moment
    update_moment_per_elem_norm
    Updates
    zero_nans
    ZeroNansState
    with_extra_args_support


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

.. autofunction:: per_example_global_norm_clip
.. autofunction:: per_example_layer_norm_clip
.. autofunction:: scale
.. autofunction:: scale_by_adam
.. autofunction:: scale_by_adamax
.. autofunction:: scale_by_amsgrad
.. autofunction:: scale_by_belief
.. autofunction:: scale_by_factored_rms
.. autofunction:: scale_by_lion
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

.. autoclass:: ScaleByLionState
    :members:

.. autoclass:: ScaleByNovogradState
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



