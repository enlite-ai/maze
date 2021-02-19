.. _runner_concept:

Runner Concept
==============

In Maze, Runners are the entity responsible for launching and administering
any job you start from a command line (like training or rollouts). They interpret
the configuration and make sure the appropriate elements (models, trainers, etc.)
are created, configured, and launched.

For a more detailed description of the runner concept,
see :ref:`Hydra overview<hydra-overview-runners>`.
If you need to write custom runners for your project, see the
:ref:`documentation for custom configuration<hydra-custom-runners>`.
