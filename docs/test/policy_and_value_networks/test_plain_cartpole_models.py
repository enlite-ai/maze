import torch
import yaml

import docs.source.policy_and_value_networks.code_snippets as code_snippets
from docs.source.policy_and_value_networks.code_snippets.custom_plain_cartpole_critic_net import \
    CustomPlainCartpoleCriticNet
from docs.source.policy_and_value_networks.code_snippets.custom_plain_cartpole_policy_net import \
    CustomPlainCartpolePolicyNet
from maze.core.utils.factory import Factory
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.perception_utils import observation_spaces_to_in_shapes


def test_cartpole_policy_model():
    env = GymMazeEnv(env='CartPole-v0')
    observation_spaces_dict = env.observation_spaces_dict
    action_spaces_dict = env.action_spaces_dict

    flat_action_space = flat_structured_space(action_spaces_dict)
    distribution_mapper = DistributionMapper(action_space=flat_action_space,
                                             distribution_mapper_config={})

    action_logits_shapes = {step_key: {action_head: distribution_mapper.required_logits_shape(action_head)
                                       for action_head in action_spaces_dict[step_key].spaces.keys()}
                            for step_key in action_spaces_dict.keys()}

    obs_shapes = observation_spaces_to_in_shapes(observation_spaces_dict)

    policy = CustomPlainCartpolePolicyNet(obs_shapes[0], action_logits_shapes[0],
                                          hidden_layer_0=16, hidden_layer_1=32, use_bias=True)

    critic = CustomPlainCartpoleCriticNet(obs_shapes[0], hidden_layer_0=16, hidden_layer_1=32, use_bias=True)

    obs_np = env.reset()
    obs = {k: torch.from_numpy(v) for k, v in obs_np.items()}

    actions = policy(obs)
    values = critic(obs)

    assert 'action' in actions
    assert 'value' in values


def test_cartpole_model_composer():
    env = GymMazeEnv(env='CartPole-v0')
    path_to_model_config = code_snippets.__path__._path[0] + '/custom_plain_cartpole_net.yaml'

    model_composer = Factory(base_type=BaseModelComposer).instantiate(
        yaml.load(open(path_to_model_config, 'r')),
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict=env.agent_counts_dict)
