from typing import Dict, Optional

from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import ActorID
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env
from maze.test.shared_test_utils.dummy_models.actor_model import DummyPolicyNet


def _dummy_model_config():
    """Model config with dummy poliy nets for two sub-steps."""
    network_config = [{"_target_": "maze.test.shared_test_utils.dummy_models.actor_model.DummyPolicyNet",
                       "non_lin": "torch.nn.SELU"}]
    return {
        "_target_": CustomModelComposer,
        "distribution_mapper_config": {},
        "policy": {
            "_target_": "maze.perception.models.policies.ProbabilisticPolicyComposer",
            "networks": network_config * 2,
            "substeps_with_separate_agent_nets": []
        },
        "critic": None
    }


def _dummy_policy_for(model_config: Dict, env: MazeEnv, agent_counts_dict: Optional[Dict] = None):
    """Helper for building a policy from env and model config, with the option to override
    the agent_counts dict returned by the env."""
    if agent_counts_dict is None:
        agent_counts_dict = env.agent_counts_dict

    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   agent_counts_dict=agent_counts_dict,
                                   distribution_mapper_config=[],
                                   policy=model_config["policy"],
                                   critic=model_config["critic"])
    return composer.policy


def test_building_shared_agent_policies():
    env = build_dummy_structured_env()
    model_config = _dummy_model_config()
    policy = _dummy_policy_for(model_config, env, agent_counts_dict={0: 1, 1: 3})

    assert len(policy.networks) == 2
    assert [0, 1] == list(policy.networks.keys())

    assert isinstance(policy.network_for(actor_id=ActorID(0, 0)), DummyPolicyNet)
    assert isinstance(policy.network_for(actor_id=ActorID(1, 0)), DummyPolicyNet)


def test_building_separated_agent_policies():
    env = build_dummy_structured_env()
    model_config = _dummy_model_config()
    model_config["policy"]["substeps_with_separate_agent_nets"] = [0, 1]
    policy = _dummy_policy_for(model_config, env, agent_counts_dict={0: 1, 1: 3})

    assert len(policy.networks) == 4
    assert [(0, 0), (1, 0), (1, 1), (1, 2)] == list(policy.networks.keys())

    assert isinstance(policy.network_for(actor_id=ActorID(0, 0)), DummyPolicyNet)
    assert isinstance(policy.network_for(actor_id=ActorID(1, 0)), DummyPolicyNet)


def test_building_separated_and_shared_agent_policies():
    env = build_dummy_structured_env()
    model_config = _dummy_model_config()
    model_config["policy"]["substeps_with_separate_agent_nets"] = [1]
    policy = _dummy_policy_for(model_config, env, agent_counts_dict={0: 2, 1: 3})

    assert len(policy.networks) == 4
    assert [0, (1, 0), (1, 1), (1, 2)] == list(policy.networks.keys())

    assert isinstance(policy.network_for(actor_id=ActorID(0, 0)), DummyPolicyNet)
    assert isinstance(policy.network_for(actor_id=ActorID(1, 0)), DummyPolicyNet)