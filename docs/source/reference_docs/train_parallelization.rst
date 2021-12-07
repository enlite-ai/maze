Parallelization
===============

This page contains the reference documentation for the parallelization module.

Vectorized Environments
------------------------

These are interfaces, classes and utility functions for vectorized environments:

.. currentmodule:: maze.train.parallelization.vector_env

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~vector_env.VectorEnv
    ~structured_vector_env.StructuredVectorEnv
    ~sequential_vector_env.SequentialVectorEnv
    ~subproc_vector_env.SubprocVectorEnv
    ~subproc_vector_env.CloudpickleWrapper
    ~vector_env_utils.SinkHoleConsumer
    ~vector_env_utils.disable_epoch_level_stats


Distributed Actors
------------------

These are interfaces, classes and utility functions for distributed actors:

.. currentmodule:: maze.train.parallelization.distributed_actors

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~distributed_actors.DistributedActors
    ~sequential_distributed_actors.SequentialDistributedActors
    ~subproc_distributed_actors.SubprocDistributedActors
    ~base_distributed_workers_with_buffer.BaseDistributedWorkersWithBuffer
    ~dummy_distributed_workers_with_buffer.DummyDistributedWorkersWithBuffer


Utilities
---------

Reusable components used in multiple distribution scenarios:

.. currentmodule:: maze.train.parallelization

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~broadcasting_container.BroadcastingContainer
    ~broadcasting_container.BroadcastingManager
