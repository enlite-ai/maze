import torch
import yaml
from gym import spaces

import docs.source.policy_and_value_networks.code_snippets as code_snippets
from docs.source.policy_and_value_networks.code_snippets.custom_complex_critic_net import \
    CustomComplexCriticNet
from docs.source.policy_and_value_networks.code_snippets.custom_complex_policy_net import \
    CustomComplexPolicyNet
from maze.core.utils.registry import Registry
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.perception_utils import observation_spaces_to_in_shapes

observation_spaces_dict = {0: spaces.Dict({'observation_inventory': spaces.Box(0, 1, shape=(8, 16,)),
                                          'observation_screen': spaces.Box(0, 1, shape=(8, 3, 64, 64))})}
action_spaces_dict = {0: spaces.Dict({'action_move': spaces.Discrete(4),
                                     'action_use': spaces.MultiBinary(16)})}


def test_cartpole_policy_model():

    flat_action_space = flat_structured_space(action_spaces_dict)
    distribution_mapper = DistributionMapper(action_space=flat_action_space,
                                             distribution_mapper_config={})

    action_logits_shapes = {step_key: {action_head: distribution_mapper.required_logits_shape(action_head)
                                       for action_head in action_spaces_dict[step_key].spaces.keys()}
                            for step_key in action_spaces_dict.keys()}

    obs_shapes = observation_spaces_to_in_shapes(observation_spaces_dict)

    policy = CustomComplexPolicyNet(obs_shapes[0], action_logits_shapes[0], non_lin='torch.nn.ReLU',
                                      hidden_units=[128])

    critic = CustomComplexCriticNet(obs_shapes[0], non_lin='torch.nn.ReLU',
                                      hidden_units=[128])

    obs_np = observation_spaces_dict[0].sample()
    obs = {k: torch.from_numpy(v) for k, v in obs_np.items()}

    actions = policy(obs)
    values = critic(obs)

    assert 'action_move' in actions
    assert 'action_use' in actions
    assert 'value' in values


def test_cartpole_model_composer():
    path_to_model_config = code_snippets.__path__._path[0] + '/custom_complex_net.yaml'

    model_composer = Registry(base_type=BaseModelComposer).arg_to_obj(
        yaml.load(open(path_to_model_config, 'r')),
        action_spaces_dict=action_spaces_dict,
        observation_spaces_dict=observation_spaces_dict)
