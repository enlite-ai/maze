.. _env_wrappers_ref:

Environment Wrappers
====================

This page contains the reference documentation for environment wrappers.
:ref:`Here <env_wrappers>` you can find a more extensive write up on how to work with these.

.. contents:: Overview
    :depth: 1
    :local:
    :backlinks: top

Interfaces and Utilities
------------------------

These are the wrapper interfaces, base classes and interfaces:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~wrapper.Wrapper

**Types of Wrappers**:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~wrapper.ObservationWrapper
    ~wrapper.ActionWrapper
    ~wrapper.RewardWrapper
    ~wrapper_registry.WrapperRegistry

Built-in Wrappers
-----------------

Below you find the reference documentation for  environment wrappers.

**General Wrappers**:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~log_stats_wrapper.LogStatsWrapper
    ~observation_logging_wrapper.ObservationLoggingWrapper
    ~time_limit_wrapper.TimeLimitWrapper
    ~random_reset_wrapper.RandomResetWrapper
    ~sorted_spaces_wrapper.SortedSpacesWrapper
    ~no_dict_spaces_wrapper.NoDictSpacesWrapper

**ObservationWrappers**:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~dict_observation_wrapper.DictObservationWrapper
    ~observation_logging_wrapper.ObservationLoggingWrapper
    ~observation_stack_wrapper.ObservationStackWrapper
    ~no_dict_observation_wrapper.NoDictObservationWrapper

**ActionWrappers**:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~dict_action_wrapper.DictActionWrapper
    ~no_dict_action_wrapper.NoDictActionWrapper
    ~split_actions_wrapper.SplitActionsWrapper
    ~discretize_actions_wrapper.DiscretizeActionsWrapper

**RewardWrappers**:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~reward_scaling_wrapper.RewardScalingWrapper
    ~reward_clipping_wrapper.RewardClippingWrapper

.. _observation_pre_processing_reference:

Observation Pre-Processing Wrapper
----------------------------------

Below you find the reference documentation for observation pre-processing.
:ref:`Here <observation_pre_processing>` you can find a more extensive write up on how to work with the
observation pre-processing package.

These are interfaces and components required for observation pre-processing:

.. currentmodule:: maze.core.wrappers.observation_preprocessing

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~preprocessing_wrapper.PreProcessingWrapper
    ~preprocessors.base.PreProcessor

These are the available built-in **maze.pre_processors** compatible with the PreProcessingWrapper:

.. currentmodule:: maze.core.wrappers.observation_preprocessing.preprocessors

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~flatten.FlattenPreProcessor
    ~one_hot.OneHotPreProcessor
    ~resize_img.ResizeImgPreProcessor
    ~transpose.TransposePreProcessor
    ~unsqueeze.UnSqueezePreProcessor
    ~rgb2gray.Rgb2GrayPreProcessor

.. _observation_normalization_reference:

Observation Normalization Wrapper
---------------------------------

Below you find the reference documentation for observation normalization.
:ref:`Here <observation_normalization>` you can find a more extensive write up on how to work with the
observation normalization package.

These are interfaces and utility functions required for observation normalization:

.. currentmodule:: maze.core.wrappers.observation_normalization

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~observation_normalization_wrapper.ObservationNormalizationWrapper
    ~normalization_strategies.base.ObservationNormalizationStrategy
    ~observation_normalization_utils.obtain_normalization_statistics
    ~observation_normalization_utils.estimate_observation_normalization_statistics
    ~observation_normalization_utils.make_normalized_env_factory

These are the available built-in **maze.normalization_strategies** compatible with the ObservationNormalizationWrapper:

.. currentmodule:: maze.core.wrappers.observation_normalization.normalization_strategies

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~mean_zero_std_one.MeanZeroStdOneObservationNormalizationStrategy
    ~range_zero_one.RangeZeroOneObservationNormalizationStrategy

.. _env_wrappers_ref-gym_env:

Gym Environment Wrapper
-----------------------

Below you find the reference documentation for wrapping gym environments.
:ref:`Here <tutorial_gym_env>` you can find a more extensive write up on how to integrate Gym environments
within Maze.

These are the contained components:

.. currentmodule:: maze.core.wrappers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~maze_gym_env_wrapper.GymMazeEnv
    ~maze_gym_env_wrapper.make_gym_maze_env
    ~maze_gym_env_wrapper.GymCoreEnv
    ~maze_gym_env_wrapper.GymRenderer
    ~maze_gym_env_wrapper.GymRewardAggregator
    ~maze_gym_env_wrapper.GymObservationConversion
    ~maze_gym_env_wrapper.GymActionConversion
