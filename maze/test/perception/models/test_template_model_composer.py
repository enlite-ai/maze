"""Contains default model tests."""
from typing import Dict, List, Union, Optional

import pytest
from torch import nn

from maze.core.agent.random_policy import RandomPolicy
from maze.core.agent.state_critic_input_output import StateCriticInput
from maze.core.agent.torch_state_critic import TorchSharedStateCritic, \
    TorchDeltaStateCritic, TorchStepStateCritic, TorchStateCritic
from maze.core.env.structured_env import StepKeyType
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.models.critics import SharedStateCriticComposer, StepStateCriticComposer, \
    BaseStateCriticComposer, DeltaStateCriticComposer
from maze.perception.models.template_model_composer import TemplateModelComposer
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env, \
    build_dummy_maze_env_with_structured_core_env

policy_composer_type = 'maze.perception.models.policies.ProbabilisticPolicyComposer'


def build_single_step_with_critic_type(critics_composer_type: type(BaseStateCriticComposer),
                                       critics_type: type(TorchStateCritic),
                                       shared_embedding_keys: Optional[Union[List[str], Dict[StepKeyType, List[str]]]]):
    """ helper function """
    # init environment
    env = GymMazeEnv('CartPole-v0')
    observation_space = env.observation_space
    action_space = env.action_space

    # map observations to a modality
    obs_modalities = {obs_key: "feature" for obs_key in observation_space.spaces.keys()}
    # define how to process a modality
    modality_config = dict()
    modality_config["feature"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                  "block_params": {"hidden_units": [32, 32],
                                                   "non_lin": "torch.nn.ReLU"}}
    modality_config["hidden"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                 "block_params": {"hidden_units": [64],
                                                  "non_lin": "torch.nn.ReLU"}}
    modality_config["recurrence"] = {}

    model_builder = {'_target_': 'maze.perception.builders.concat.ConcatModelBuilder',
                     'modality_config': modality_config,
                     'observation_modality_mapping': obs_modalities,
                     'shared_embedding_keys': shared_embedding_keys}

    # initialize default model builder
    default_builder = TemplateModelComposer(action_spaces_dict={0: action_space},
                                            observation_spaces_dict={0: observation_space},
                                            agent_counts_dict={0: 1},
                                            distribution_mapper_config={},
                                            model_builder=model_builder,
                                            policy={'_target_': policy_composer_type},
                                            critic={'_target_': critics_composer_type})

    # create model pdf
    default_builder.save_models()

    assert isinstance(default_builder.distribution_mapper, DistributionMapper)
    assert isinstance(default_builder.policy.networks[0], nn.Module)
    assert isinstance(default_builder.critic.networks[0], nn.Module)
    assert isinstance(default_builder.critic, critics_type)

    # test default policy gradient actor
    policy_net = default_builder.policy.networks[0]
    assert isinstance(policy_net, InferenceBlock)

    assert "action" in policy_net.out_keys
    assert policy_net.out_shapes()[0] == (2,)

    # test standalone critic
    value_net = default_builder.critic.networks[0]
    assert isinstance(value_net, InferenceBlock)
    assert "value" in value_net.out_keys
    assert value_net.out_shapes()[0] == (1,)

    if shared_embedding_keys is not None:
        if isinstance(shared_embedding_keys, list):
            assert all([shared_key in policy_net.out_keys for shared_key in shared_embedding_keys])
            assert all([shared_key in value_net.in_keys for shared_key in shared_embedding_keys])
        else:
            assert all([shared_key in policy_net.out_keys for shared_keylist in shared_embedding_keys.values() for
                        shared_key in shared_keylist])
            assert all([shared_key in value_net.in_keys for shared_keylist in shared_embedding_keys.values() for
                        shared_key in shared_keylist])
    else:
        assert value_net.in_keys == policy_net.in_keys

    rollout_generator = RolloutGenerator(env=env, record_next_observations=False)
    policy = RandomPolicy(env.action_spaces_dict)
    trajectory = rollout_generator.rollout(policy, n_steps=10).stack().to_torch(device='cpu')

    policy_output = default_builder.policy.compute_policy_output(trajectory)
    critic_input = StateCriticInput.build(policy_output, trajectory)
    _ = default_builder.critic.predict_values(critic_input)


def test_default_models() -> None:
    """ default model builder test. """
    build_single_step_with_critic_type(SharedStateCriticComposer, TorchSharedStateCritic,
                                       shared_embedding_keys=None)
    build_single_step_with_critic_type(StepStateCriticComposer, TorchStepStateCritic,
                                       shared_embedding_keys=None)
    build_single_step_with_critic_type(DeltaStateCriticComposer, TorchDeltaStateCritic,
                                       shared_embedding_keys=None)

    build_single_step_with_critic_type(StepStateCriticComposer, TorchStepStateCritic,
                                       shared_embedding_keys=['latent'])
    build_single_step_with_critic_type(StepStateCriticComposer, TorchStepStateCritic,
                                       shared_embedding_keys={0: ['observation_DenseBlock']})


def build_structured_with_critic_type(env,
                                      critics_composer_type: type(BaseStateCriticComposer),
                                      critics_type: type(TorchStateCritic),
                                      shared_embedding_keys: Optional[Union[List[str], Dict[StepKeyType, List[str]]]]):
    """ helper function """

    # map observations to a modality
    obs_modalities = {"observation_0": "image",
                      "observation_1": "feature",
                      DeltaStateCriticComposer.prev_value_key: 'feature'}

    # define how to process a modality
    modality_config = dict()
    modality_config["feature"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                  "block_params": {"hidden_units": [32, 32],
                                                   "non_lin": "torch.nn.ReLU"}}
    modality_config['image'] = {'block_type': 'maze.perception.blocks.StridedConvolutionDenseBlock',
                                'block_params': {'hidden_channels': [8, 16, 32],
                                                 'hidden_kernels': [8, 4, 4],
                                                 'convolution_dimension': 2,
                                                 'hidden_strides': [4, 2, 2],
                                                 'hidden_dilations': None,
                                                 'hidden_padding': [1, 1, 1],
                                                 'padding_mode': None,
                                                 'hidden_units': [],
                                                 'non_lin': 'torch.nn.SELU'}}

    modality_config["hidden"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                 "block_params": {"hidden_units": [64],
                                                  "non_lin": "torch.nn.ReLU"}}
    modality_config["recurrence"] = {}

    model_builder = {'_target_': 'maze.perception.builders.concat.ConcatModelBuilder',
                     'modality_config': modality_config,
                     'observation_modality_mapping': obs_modalities,
                     'shared_embedding_keys': shared_embedding_keys}

    # initialize default model builder
    default_builder = TemplateModelComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict=env.agent_counts_dict,
        distribution_mapper_config={},
        model_builder=model_builder,
        policy={'_target_': policy_composer_type},
        critic={'_target_': critics_composer_type})

    # create model pdf
    default_builder.save_models()

    assert isinstance(default_builder.distribution_mapper, DistributionMapper)
    for pp in default_builder.policy.networks.values():
        assert isinstance(pp, nn.Module)
    for cc in default_builder.critic.networks.values():
        assert isinstance(cc, nn.Module)

    assert isinstance(default_builder.critic, critics_type)

    rollout_generator = RolloutGenerator(env=env, record_next_observations=False)
    policy = RandomPolicy(env.action_spaces_dict)
    trajectory = rollout_generator.rollout(policy, n_steps=10).stack().to_torch(device='cpu')

    policy_output = default_builder.policy.compute_policy_output(trajectory)
    critic_input = StateCriticInput.build(policy_output, trajectory)
    _ = default_builder.critic.predict_values(critic_input)


def test_default_models_multi_step() -> None:
    """ default model builder test. """
    env = build_dummy_structured_env()
    build_structured_with_critic_type(env, SharedStateCriticComposer, TorchSharedStateCritic,
                                      shared_embedding_keys=None)
    build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic, shared_embedding_keys=None)
    build_structured_with_critic_type(env, DeltaStateCriticComposer, TorchDeltaStateCritic, shared_embedding_keys=None)


def test_default_shared_model_multi_obs() -> None:
    env = build_dummy_maze_env_with_structured_core_env()
    build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic,
                                      shared_embedding_keys=['latent'])
    build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic,
                                      shared_embedding_keys=['observation_0_StridedConvolutionDenseBlock',
                                                             'observation_1_DenseBlock'])
    with pytest.raises(AssertionError):
        build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic,
                                          shared_embedding_keys=['concat',
                                                                 'observation_1'])


def test_default_shared_models_multi_step() -> None:
    env = build_dummy_structured_env()
    build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic,
                                      shared_embedding_keys=['latent'])
    build_structured_with_critic_type(env, StepStateCriticComposer, TorchStepStateCritic,
                                      shared_embedding_keys={0: ['latent'], 1: ['observation_1_DenseBlock']})
