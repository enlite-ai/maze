.. _struct_env_flat:

Flat Environments
=================

.. note::
    Recommended reads prior to this article:
        - :ref:`Control Flows with Structured Environments<control_flows_struct_envs>`.

All instantiable environments in Maze are subclasses of :class:`StructuredEnv <maze.core.env.structured_env.StructuredEnv>`. Structured environments are discussed in :ref:`Control Flows with Structured Environments<control_flows_struct_envs>`, which we recommend to read prior to this article. *Flat* environments in our terminology are those utilizing a single actor and a single policy, i. e. a single actor, and conducting one action per step. Within Maze, flat environments are a special case of structured environments.

An exemplary implementation of a flat environment for the `stock cutting problem <https://en.wikipedia.org/wiki/Cutting_stock_problem>`_ can be found :ref:`here<env_from_scratch-problem>`.

Control Flow
------------

Let's revisit a classic depiction of a RL control flow first:

.. figure:: control_flow_simple.png
    :width: 80 %
    :align: center

    Simplified control flow within a flat scenario. The agent selects an action, the environment updates its state and computes the reward. There is no need to distinguish between different policies or agents since we only have one of each. :meth:`~maze.core.env.structured_env.StructuredEnv.actor_id` should always return the same value.

A more general framework however needs to be able to integrate multiple agents and policies into its control flow. Maze does this by implementing actors, which are abstractions introduced in the RL literature to represent one policy applied on or used by one agent.
The figure above collapses the concepts of policy, agent and actor into a single entity for the sake of simplicity. The actual control flow for a flat environment in Maze is closer to this:

.. figure:: control_flow_complex.png
    :width: 80 %
    :align: center

    More accurate control flow for a flat environment in Maze, showing how the actor mechanism integrates agent and policy. Dashed lines denote the exchange of information on demand as opposed to doing so passing it to or returning it from :meth:`~maze.core.env.maze_env.MazeEnv.step`.

A flat environment hence always utilizes the same actor, i.e. the same policy for the same agent. Due to the lack of other actors there is no need for the environment to ever update its active actor ID.
The concept of actors is crucial to the flexibility of Maze, since it allows to scale up the number of agents, policies or both. This enables the application of RL to a wider range of use cases and exploit properties of the respective domains more efficiently.

Where to Go Next
----------------

- :ref:`Multi-stepping applies the actor mechanism to enact several policies in a single step<struct_env_multistep>`.
- :ref:`Multi-agent RL by using multiple actors with different agents<struct_env_multiagent>`.
- :ref:`Hierarchical RL by chaining and nesting tasks via policies<struct_env_hierarchical>`.