""" Contains unit tests for custom model composers. """
import os

from maze.core.agent.torch_state_critic import TorchSharedStateCritic, \
    TorchDeltaStateCritic, TorchStepStateCritic
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.critics import DeltaStateCriticComposer
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.perception.perception_utils import convert_to_torch
from maze.test.shared_test_utils.dummy_models.critic_model import DummyValueNet
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env


def test_custom_model_composer():
    """ Perception unit tests """

    # init structured env
    env = build_dummy_structured_env()

    policies = {
        "_target_": "maze.perception.models.policies.ProbabilisticPolicyComposer",
        "networks": [{"_target_": "maze.test.shared_test_utils.dummy_models.actor_model.DummyPolicyNet",
                      "non_lin": "torch.nn.SELU"},
                     {"_target_": "maze.test.shared_test_utils.dummy_models.actor_model.DummyPolicyNet",
                      "non_lin": "torch.nn.SELU"}],
        "substeps_with_separate_agent_nets": []
    }

    # check if model config is fine
    CustomModelComposer.check_model_config({"policy": policies})

    # no critic
    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   agent_counts_dict=env.agent_counts_dict,
                                   distribution_mapper_config=[],
                                   policy=policies,
                                   critic=None)

    assert isinstance(composer.distribution_mapper, DistributionMapper)
    assert composer.critic is None

    # shared critic
    shared_critic = {
        "_target_": "maze.perception.models.critics.SharedStateCriticComposer",
        "networks": [{"_target_": "maze.test.shared_test_utils.dummy_models.critic_model.DummyValueNet",
                      "non_lin": "torch.nn.SELU"}],
        "stack_observations": False
    }

    # check if model config is fine
    CustomModelComposer.check_model_config({"critic": shared_critic})

    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   agent_counts_dict=env.agent_counts_dict,
                                   distribution_mapper_config=[],
                                   policy=policies,
                                   critic=shared_critic)

    assert isinstance(composer.distribution_mapper, DistributionMapper)
    assert isinstance(composer.critic, TorchSharedStateCritic)
    assert isinstance(composer.critic.networks, dict)
    assert isinstance(composer.critic.networks[0], DummyValueNet)

    # delta critic
    step_critic = {
        "_target_": "maze.perception.models.critics.DeltaStateCriticComposer",
        "networks": [
            {"_target_": "maze.test.shared_test_utils.dummy_models.critic_model.DummyValueNet",
             "non_lin": "torch.nn.SELU"},
            {"_target_": "maze.test.shared_test_utils.dummy_models.critic_model.DummyValueNet",
             "non_lin": "torch.nn.SELU"}
        ]
    }

    # check if model config is fine
    CustomModelComposer.check_model_config({"critic": step_critic})

    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   agent_counts_dict=env.agent_counts_dict,
                                   distribution_mapper_config=[],
                                   policy=policies,
                                   critic=step_critic)

    assert isinstance(composer.distribution_mapper, DistributionMapper)
    assert isinstance(composer.critic, TorchDeltaStateCritic)
    assert isinstance(composer.critic.networks, dict)
    assert isinstance(composer.critic.networks[0], DummyValueNet)
    assert isinstance(composer.critic.networks[1], DummyValueNet)
    value_0 = composer.critic.networks[0](
        convert_to_torch(env.observation_spaces_dict[0].sample(), device=None, cast=None,
                         in_place=True))
    value_1 = composer.critic.networks[1](
        {**convert_to_torch(env.observation_spaces_dict[1].sample(), device=None, cast=None,
                            in_place=True),
         DeltaStateCriticComposer.prev_value_key: value_0['value']})

    composer.save_models()

    # step critic
    step_critic = {
        "_target_": "maze.perception.models.critics.StepStateCriticComposer",
        "networks": [
            {"_target_": "maze.test.shared_test_utils.dummy_models.critic_model.DummyValueNet",
             "non_lin": "torch.nn.SELU"},
            {"_target_": "maze.test.shared_test_utils.dummy_models.critic_model.DummyValueNet",
             "non_lin": "torch.nn.SELU"}
        ]
    }

    # check if model config is fine
    CustomModelComposer.check_model_config({"critic": step_critic})

    composer = CustomModelComposer(action_spaces_dict=env.action_spaces_dict,
                                   observation_spaces_dict=env.observation_spaces_dict,
                                   agent_counts_dict=env.agent_counts_dict,
                                   distribution_mapper_config=[],
                                   policy=policies,
                                   critic=step_critic)

    assert isinstance(composer.distribution_mapper, DistributionMapper)
    assert isinstance(composer.critic, TorchStepStateCritic)
    assert isinstance(composer.critic.networks, dict)
    assert isinstance(composer.critic.networks[0], DummyValueNet)
    assert isinstance(composer.critic.networks[1], DummyValueNet)

    # test saving models
    composer.save_models()

    try:
        import pygraphviz

        for model_file in ["critic_0.pdf", "critic_1.pdf", "policy_0.pdf", "policy_1.pdf"]:
            file_path = os.path.join(os.getcwd(), model_file)
            assert os.path.exists(file_path)
            os.remove(file_path)
    except ImportError as e:
        pass  # no output generated as pygraphviz is not installed.
