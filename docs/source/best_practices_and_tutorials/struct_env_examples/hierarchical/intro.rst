.. _struct_env_hierarchical:

Hierarchical RL with Structured Environments
============================================

.. note::
    Recommended reads prior to this article:
        - :ref:`Control Flows with Structured Environments<control_flows_struct_envs>`.
        - :ref:`Flat Environments as a special case of structured environments<control_flows_struct_envs>`.

Hiearchical reinforcement learning


todos
- mention connection to action masking
- (potential) advantages over action masking: cleaner code, no learning of masking effect required (?), reusable higher-level
- example: fabriation of component out of parts?
- https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
-> reduce action space, generalization/modularity by transfering meta-policies to similar problems, decomposition into task hierarchy is reflected in cleaner code (action masking not - or less - required, received actions are only containing relevant action subspaces).

Where to Go Next
----------------

- :ref:`Gym-style flat environments as a special case of structured environments<struct_env_multiagent>`.
- :ref:`Multi-stepping applies the actor mechanism to enact several policies in a single step<struct_env_multistep>`.
- :ref:`Multi-agent RL by using multiple actors with different agents<struct_env_multiagent>`.
- :ref:`Arbitrary environments with evolutionary strategies<struct_env_evolutionary>` [todo].