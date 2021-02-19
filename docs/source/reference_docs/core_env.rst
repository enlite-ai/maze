.. _environment_interfaces_reference:

Environment Interfaces
======================

This page contains the reference documentation for environment interfaces.

maze.core.env
-------------

Environment interfaces:

.. currentmodule:: maze.core.env

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~base_env.BaseEnv
    ~structured_env.StructuredEnv
    ~core_env.CoreEnv
    ~structured_env_spaces_mixin.StructuredEnvSpacesMixin
    ~maze_env.MazeEnv
    ~render_env_mixin.RenderEnvMixin
    ~recordable_env_mixin.RecordableEnvMixin
    ~serializable_env_mixin.SerializableEnvMixin
    ~time_env_mixin.TimeEnvMixin
    ~event_env_mixin.EventEnvMixin
    ~simulated_env_mixin.SimulatedEnvMixin

Interfaces for additional components:

.. currentmodule:: maze.core.env

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~observation_conversion.ObservationConversionInterface
    ~action_conversion.ActionConversionInterface
    ~maze_state.MazeStateType
    ~maze_action.MazeActionType
    ~reward.RewardAggregatorInterface
    ~environment_context.EnvironmentContext
