General and Rollout Runners
===========================

This page contains the reference documentation for all kinds of runners.

.. contents:: Overview
    :depth: 1
    :local:
    :backlinks: top

General Runners
---------------

These are the basic interfaces, classes and utility functions of runners:

.. currentmodule:: maze

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~runner.Runner
    ~maze_cli.maze_run

Rollout Runners
---------------

These are interfaces, classes and utility functions for rollout runners:

:ref:`Here <train_ref>` can find the documentation for training runners.

.. currentmodule:: maze.core.rollout

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~rollout_runner.RolloutRunner
    ~rollout_generator.RolloutGenerator
    ~sequential_rollout_runner.SequentialRolloutRunner
    ~parallel_rollout_runner.ParallelRolloutRunner
    ~parallel_rollout_runner.ParallelRolloutWorker
    ~parallel_rollout_runner.EpisodeRecorder
    ~parallel_rollout_runner.EpisodeStatsReport
    ~parallel_rollout_runner.ExceptionReport
