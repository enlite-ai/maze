"""Contains unit tests for the serialized torch policy."""
import torch

from maze.core.agent.serialized_torch_policy import SerializedTorchPolicy
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.perception.models.space_config import SpacesConfig
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env


def test_serialized_torch_policy():
    """ unit tests """

    # init structured env
    env = build_dummy_structured_env()

    model_config = {
        "type": CustomModelComposer,
        "distribution_mapper_config": {},
        "policy": {
            "type": "maze.perception.models.policies.ProbabilisticPolicyComposer",
            "networks": [{"type": "maze.test.shared_test_utils.dummy_models.actor_model.DummyPolicyNet",
                          "non_lin": "torch.nn.SELU"},
                         {"type": "maze.test.shared_test_utils.dummy_models.actor_model.DummyPolicyNet",
                          "non_lin": "torch.nn.SELU"}]
        },
        "critic": None
    }

    # no critic
    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   distribution_mapper_config=[],
                                   policy=model_config["policy"],
                                   critic=model_config["critic"])

    # dump state dict
    state_dict = composer.policy.state_dict()
    torch.save(state_dict, "state_dict.pt")

    SpacesConfig(composer.action_spaces_dict,
                 composer.observation_spaces_dict).save("spaces_config.pkl")

    # init policy
    policy = SerializedTorchPolicy(model=model_config, state_dict_file="state_dict.pt",
                                   spaces_dict_file="spaces_config.pkl", device="cpu")

    action = policy.compute_action(observation=env.observation_space.sample(), policy_id=0)
    assert isinstance(action, dict)
