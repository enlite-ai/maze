.. _control_flows_struct_envs:

Control Flows with Structured Environments
==========================================

The basic reinforcement learning formulation assumes a single actor in an environment, enacting one policy-suggested
action at a step. A classic example for this is the cartpole balancing problem, in which a single actor attempts to
balance a cartpole as stably as possible. However, some problems incentivize or even require these assumptions to be
violated:
- *Single actor*: Plenty of real-world scenarios motivate taking several actors into account. Imagine e.g. trying to
optimize the coordination of a fleet of delivery vehicles - clearly there are emergent effects and interdependences
between individual vehicles, such as that the availability and suitability of orders for any given vehicle depends on
the proximity and activity of other vehicles. Treating them in isolation from each other is inefficient and will lead to
inferior results.
* *One action at a time*: Some usecases necessarily involve a sequence of actions. (todo: mention relation to
autoregressive actions, cutting 2D as example)

Examples:

:ref:`Structured Auto-Regressive Environments<struct_env_autoregressive>`
:ref:`Structured Multi-Agent Environments<struct_env_multiagent>`
:ref:`Structured Hierarchical Environments<struct_env_hierarchical>`
:ref:`Structured Environments with After-States<struct_env_afterstate>`
:ref:`Structured Environments with Evolutionary Strategies<struct_env_evolutionary>`
