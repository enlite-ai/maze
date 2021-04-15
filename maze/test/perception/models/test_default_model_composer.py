"""Contains default model tests."""
from maze.core.agent.torch_state_critic import TorchSharedStateCritic, \
    TorchDeltaStateCritic, TorchStepStateCritic, TorchStateCritic
from torch import nn

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.critics import SharedStateCriticComposer, StepStateCriticComposer, BaseStateCriticComposer, \
    DeltaStateCriticComposer
from maze.perception.models.template_model_composer import TemplateModelComposer
from maze.perception.perception_utils import convert_to_torch, flatten_spaces
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env

policy_composer_type = 'maze.perception.models.policies.ProbabilisticPolicyComposer'


def build_single_step_with_critic_type(critics_composer_type: type(BaseStateCriticComposer),
                                       critics_type: type(TorchStateCritic)):
    """ helper function """
    # init environment
    env = GymMazeEnv(env="CartPole-v0")
    observation_space = env.observation_space
    action_space = env.action_space

    # map observations to a modality
    obs_modalities = {"observation": "feature"}

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
                     'observation_modality_mapping': obs_modalities}

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
    policy_net = default_builder.template_policy_net(observation_space=observation_space, action_space=action_space)
    assert "action" in policy_net.out_keys
    assert policy_net.out_shapes()[0] == (2,)

    # test standalone critic
    value_net = default_builder.template_value_net(observation_space=observation_space)
    assert "value" in value_net.out_keys
    assert value_net.out_shapes()[0] == (1,)

    # test shared critic
    value_net = default_builder.template_value_net(observation_space=None, perception_net=policy_net)
    assert "value" in value_net.out_keys
    assert value_net.out_shapes()[0] == (1,)


def test_default_models() -> None:
    """ default model builder test. """
    build_single_step_with_critic_type(SharedStateCriticComposer, TorchSharedStateCritic)
    build_single_step_with_critic_type(StepStateCriticComposer, TorchStepStateCritic)
    build_single_step_with_critic_type(DeltaStateCriticComposer, TorchDeltaStateCritic)


def build_structured_with_critic_type(critics_composer_type: type(BaseStateCriticComposer),
                                      critics_type: type(TorchStateCritic)):
    """ helper function """
    # init environment
    env = build_dummy_structured_env()

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
                                                 'hidden_padding': None,
                                                 'padding_mode': None,
                                                 'hidden_units': [],
                                                 'non_lin': 'torch.nn.SELU'}}

    modality_config["hidden"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                 "block_params": {"hidden_units": [64],
                                                  "non_lin": "torch.nn.ReLU"}}
    modality_config["recurrence"] = {}

    model_builder = {'_target_': 'maze.perception.builders.concat.ConcatModelBuilder',
                     'modality_config': modality_config,
                     'observation_modality_mapping': obs_modalities}

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

    for step_key, step_obs in env.observation_spaces_dict.items():
        torch_obs = convert_to_torch(step_obs.sample(), cast=None, device=None, in_place=True)
        logits = default_builder.policy.networks[step_key](torch_obs)

    obs_t = {step_key: convert_to_torch(step_obs.sample(), cast=None, device=None, in_place=True)
             for step_key, step_obs in env.observation_spaces_dict.items()}
    if critics_composer_type == SharedStateCriticComposer:
        flattened_obs_t = flatten_spaces(obs_t.values())
        value = default_builder.critic.networks[0](flattened_obs_t)["value"][..., 0]
    elif critics_composer_type == StepStateCriticComposer:
        for step_id in env.observation_spaces_dict.keys():
            value = default_builder.critic.networks[step_id](obs_t[step_id])["value"][..., 0]
    elif critics_composer_type == DeltaStateCriticComposer:
        value_0 = default_builder.critic.networks[0](obs_t[0])["value"][..., 0]
        obs_t[1].update({DeltaStateCriticComposer.prev_value_key: value_0.unsqueeze(-1)})
        value_1 = default_builder.critic.networks[1](obs_t[1])["value"][..., 0]
    else:
        raise ValueError("Invalid CriticType <{}> selected!".format(critics_composer_type))


def test_default_models_multi_step() -> None:
    """ default model builder test. """
    build_structured_with_critic_type(SharedStateCriticComposer, TorchSharedStateCritic)
    build_structured_with_critic_type(StepStateCriticComposer, TorchStepStateCritic)
    build_structured_with_critic_type(DeltaStateCriticComposer, TorchDeltaStateCritic)
