.. _struct_env_multiagent:

Multi-Agent RL with Structured Environments
===========================================

`Multi-agent reinforcment learning (MARL) <https://arxiv.org/abs/1911.10635>`_ describes a setup in which more than several collaborating or competing agents act as individual entitites in an environment. This introduces the additional complexity of emergent effects between those agents. Some problems require to or at least benefit from deviating from a single-agent formulation, such as the `vehicle routing problem <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_, `(video) games like Starcraft <https://www.nature.com/articles/s41586-019-1724-z>`_, `traffic coordination<http://www.wiomax.com/team/xie/paper/ICAPS12.pdf>`_, `power systems and smart grids<https://ieeexplore.ieee.org/abstract/document/7855760>`_ and many others.

Maze supports multi-agent learning out of the box. In order to make a :class:`StructuredEnv <maze.core.env.structured_env.StructuredEnv>` compatible with such a setup, it needs to keep track of the activities of each individual agent. While the order in which actions for the individual agents are enacted are left to the environment, it is required that at any point in time there is exactly one active actor (see :ref:`here<control_flows_struct_envs>` for more information on the distinction between actor and agent). It is easily possible, but not necessary, to include multiple policies in a multi-agent scenario.

Information on the active actor and therefore agent is accessed via :meth:`~maze.core.env.structured_env.StructuredEnv.actor_id`. As long as the environment implements this method and the internal state of each agent correctly, there are no further prerequisites to be fulfilled for a multi-agent training.



    Control flow within a multi-agent scenario. Note that we assume a single policy here. In this setup :meth:`~maze.core.env.structured_env.StructuredEnv.actor_id` should return a tuple of the current actor ID and a constant policy key.

You'll notice that the actor entity coordinating policies and agents to compute an appropriate action.
In comparing with the flat environment you'll notice that this flow differs in (a) that there are multiple agents, (b) that the agent is chosen according to the active actor ID specified by the environment and (c) that the environment's step function has to do keep track on the active agent/actor. The underlying pathways however are identical - within Maze' actor mechanism flat environments are merely a particular specification amongst many.

.. toctree::
   :maxdepth: 1
   :hidden:

   test.rst