Parallelization
===============

This page contains the reference documentation for the parallelization module.

.. currentmodule:: maze.train.parallelization

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~observation_aggregator.ObservationAggregator

    ~base_worker.BaseWorker
    ~base_worker.BaseWorkerOutput

Distributed Environments
------------------------

These are interfaces, classes and utility functions for distributed environments:

.. currentmodule:: maze.train.parallelization.distributed_env

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~distributed_env.BaseDistributedEnv
    ~dummy_distributed_env.DummyStructuredDistributedEnv
    ~subproc_distributed_env.SubprocStructuredDistributedEnv
    ~subproc_distributed_env.SinkHoleConsumer
    ~subproc_distributed_env.CloudpickleWrapper

Distributed Actors
------------------

These are interfaces, classes and utility functions for distributed actors:

.. currentmodule:: maze.train.parallelization.distributed_actors

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~actor.ActorAgent
    ~distributed_actors.BaseDistributedActors
    ~dummy_distributed_actors.DummyDistributedActors
    ~subproc_distributed_actors.SubprocDistributedActors
    ~subproc_distributed_actors.MyManager
    ~broadcasting_container.BroadcastingContainer
