.. _control_flows_struct_envs:

Control Flows with Structured Environments
==========================================

The basic reinforcement learning formulation assumes a single actor in an environment, enacting one policy-suggested action per step to fulfill exactly one task. We refer to this as a *flat* environment. A classic example for this is the cartpole balancing problem, in which a single actor attempts to balance a cartpole as stably as possible. However, some problems incentivize or even require these assumptions to be violated:

 #. *Single actor*: Plenty of real-world scenarios motivate taking several actors into account. E.g.: `optimizing delivery with a fleet of vehicles <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_ involves emergent effects and interdependences between individual vehicles, such as that the availability and suitability of orders for any given vehicle depends on the proximity and activity of other vehicles. Treating them in isolation from each other is inefficient and detrimental to the learning process.
 #. *One action per step*: Some usecases, such as `cutting raw material according to customer specifications with as little waste as possible <https://en.wikipedia.org/wiki/Cutting_stock_problem>`_, necessarily involve a well-defined sequence of actions. Stock-cutting involves (a) the selection of a piece of suitable size and (b) cutting it in an appropriate manner. We know that (a) is always followed by (b) and that the latter is a necessary precondition for the former. We can incorporate this information in our RL control loop to facilitate a faster learning process by enforcing that the environment should always execute two actions in a single step: First select, then cut.
 #. *Exactly one task*: Occasionally, the problem we want to solve cannot be neatly formulated as a single task, but consists of a hierarchy of tasks. This is exemplified by `pick and place robots <https://6river.com/what-is-a-pick-and-place-robot/>`_. They solve a complex task, which is reflected by the associated hierarchy of goals: The overall goal requires (a) reaching the target object, (b) grasping the target object, (c) moving target object to target location and (d) placing the target object safey in the target location. Solving this task cannot be reduced to a single goal.

.. _control_flows_struct_envs_approach:

Beyond Flat Environments with Actors and Tasks
----------------------------------------------

Maze addresses the first assumption *(A1)* by associating actions to actors instead of the environment per se, *(A2)* by decoupling actions from steps and *(A3)* by allowing environments to specify which task is currently active.
This is supported out-of-the-box and baked into Maze' control flow with a single mechanism: Every :class:`StructuredEnv <maze.core.env.structured_env.StructuredEnv>` is required to implement :meth:`~maze.core.env.structured_env.StructuredEnv.actor_id`, which returns the ID of the currently active task and the numeric index of this actor. A single-actor, single-task, single-action-per-step environment always return (0, 0).

The environment determines the active task and actor based on its internal state. The policy used to select actions based on the available state is determined by the actor ID, i.e. multiple actor IDs entail multiple policies being used in the training process. The policy associated with the current actor evaluates the observation provided by the environment and selects an appropriate action. This action updates the environment's state, after which the role of the active actor is reevaluated and potentially reassigned.

Since different tasks may benefit from or even depend on a different preprocessing of their actions and/or observations (especially, but not exclusively, action masking), Maze requires the specification of a corresponding :class:`ActionConversionInterface <maze.core.env.action_conversion.ActionConversionInterface>` and :class:`ObservationConversionInterface <maze.core.env.observation_conversion.ObservationConversionInterface>` classes for each task.

Maze Mechanisms in a Broader Context
------------------------------------

Assumptions *(A1)* and *(A3)* are related to concepts well established in literature, namely `multi-agent learning <https://arxiv.org/abs/1911.10635>`_ and `hierarchical RL <https://arxiv.org/abs/1909.10618>`_, both of which are supported by Maze.
The problem underlying *(A2)* is a lack of temporal coherency in the sequence of selected actions: if there is some necessary, recurring order of actions, we would like to identify it as quickly as possible. We provide two different mechanisms to tackle this:

- `Auto-regressive action distributions (ARAD) <https://docs.ray.io/en/master/rllib-models.html#autoregressive-action-distributions>`_. ARADs still execute one action per step, but condition it on the previous state and *action* instead of the state alone. This allows it to be more sensitive  towards such recurring patterns of actions.
- *Multi-stepping*. This is a pattern that utilizes the actor-task mechanism to enact multiple sub-steps in their correct order and a single step without having to rely on autoregressive policies.

Both approaches aim at increasing the temporal coherence of actions. Multi-stepping allows to incorporate domain knowledge and can be used to imitate ARAD, but depends on a fixed definition of substeps in an environment's step function. ARAD does not require and cannot make use of any prior domain knowledge w.r.t. the desired step order and thus needs to learn it from data. When facing a decision on which one to use, we recommend multi-stepping if possible to exploit available domain knowledge and increase training efficiency.


todo:

- charts
- describe and link to examples
- look for available examples (flat env, autoregressive/multistep, ?)

.. _control_flows_struct_envs_next:

Where to Go Next
----------------

.. _control_flows_struct_envs_examples:

**Examples**

We provide a set of exemplary environments implementing :class:`StructuredEnv <maze.core.env.structured_env.StructuredEnv>` with different configurations:

- A detailled walkthrough for a :ref:`flat environment<struct_env_autoregressive>` not utilizing the actor-task mechanism for the stock cutting problem.
- A :ref:`structured auto-regressive environment<struct_env_autoregressive>` using a auto-regressive policy for temporally more coherent actions.
- A :ref:`structured multi-agent environment<struct_env_multiagent>` for the coordination of a fleet of delivery vehicles utilizing a set of actors.
- A :ref:`structured hierarchical environment<struct_env_hierarchical>` representing a robotic arm picking and placing object, iterating over a sequence of sub-goals.
- A :ref:`structured environment with after-states<struct_env_afterstate>` [todo].
- A :ref:`structured environments with evolutionary etrategies<struct_env_evolutionary>` [todo].

**todo**