.. _policies_and_agents_ref:

Policies, Critics and Agents
============================

This page contains the reference documentation for policies, critics and agents.

maze.core.agent
---------------

**Policies:**

.. currentmodule:: maze.core.agent

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~flat_policy.FlatPolicy
    ~policy.Policy
    ~torch_policy.TorchPolicy

    ~default_policy.DefaultPolicy
    ~random_policy.RandomPolicy
    ~dummy_cartpole_policy.DummyCartPolePolicy

    ~serialized_torch_policy.SerializedTorchPolicy

**Critics:**

.. currentmodule:: maze.core.agent

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~state_critic.StateCritic
    ~torch_state_critic.TorchStateCritic
    ~torch_state_critic.TorchSharedStateCritic
    ~torch_state_critic.TorchStepStateCritic
    ~torch_state_critic.TorchDeltaStateCritic

    ~state_action_critic.StateActionCritic
    ~torch_state_action_critic.TorchStateActionCritic
    ~torch_state_action_critic.TorchSharedStateActionCritic
    ~torch_state_action_critic.TorchStepStateActionCritic


**Models:**

.. currentmodule:: maze.core.agent

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~torch_model.TorchModel
    ~torch_actor_critic.TorchActorCritic
