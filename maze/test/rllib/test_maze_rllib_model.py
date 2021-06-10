"""test maze rllib model"""
import os
import random
from typing import Union

import numpy as np
import pytest
import torch
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn

from maze.core.utils.factory import Factory
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.perception_utils import convert_to_torch
from maze.rllib.maze_rllib_action_distribution import MazeRLlibActionDistribution
from maze.rllib.maze_rllib_models.maze_rllib_ac_model import MazeRLlibACModel
from maze.rllib.maze_rllib_models.maze_rllib_base_model import MazeRLlibBaseModel
from maze.rllib.maze_rllib_models.maze_rllib_policy_model import MazeRLlibPolicyModel
from maze.rllib.maze_rllib_models.maze_rllib_q_model import MazeRLlibQModel

SPACE_CONFIG_DUMP_FILE = 'space_config_dump.pkl'


def build_default_cartpole_model(gym_env: str, maze_rllib_model_cls: Union[type(MazeRLlibBaseModel),
                                                                           type(TorchModelV2)],
                                 add_critic: bool):
    """ helper function """
    # init environment
    env = GymMazeEnv(env=gym_env)
    observation_space = env.observation_space
    action_space = env.action_space

    model_composer_config = dict()
    model_composer_config['_target_'] = 'maze.perception.models.template_model_composer.TemplateModelComposer'
    model_composer_config['distribution_mapper_config'] = {}
    model_composer_config['model_builder'] = {
        '_target_': 'maze.perception.builders.ConcatModelBuilder',
        'observation_modality_mapping': {'observation': 'feature'},
        'modality_config': {
            'feature': {
                'block_type': 'maze.perception.blocks.DenseBlock',
                'block_params': {
                    'hidden_units': [8],
                    'non_lin': 'torch.nn.SELU'
                }
            },
            'hidden': {},
            'recurrence': {}
        }
    }
    model_composer_config['policy'] = {'_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer'}
    model_composer_config['critic'] = {'_target_': 'maze.perception.models.critics.StateCriticComposer'} \
        if add_critic else None

    class WrappedObsSpace:

        def __init__(self, maze_obs_space):
            self.original_space = maze_obs_space

    obs_space = WrappedObsSpace(observation_space)
    model_config = {'custom_model_config': {
        'maze_model_composer_config':
            {'distribution_mapper_config': model_composer_config['distribution_mapper_config']}
    },
        'vf_share_layers': False}
    num_outputs = MazeRLlibActionDistribution.required_model_output_shape(action_space, model_config)
    reset_seed()
    rllib_model = \
        maze_rllib_model_cls(obs_space,
                             action_space['action'] if issubclass(maze_rllib_model_cls, MazeRLlibQModel)
                             else action_space, num_outputs, model_config, 'test_model',
                             model_composer_config, SPACE_CONFIG_DUMP_FILE, 'state_dict.pt')
    reset_seed()
    maze_model_composer = Factory(base_type=BaseModelComposer).instantiate(
        model_composer_config,
        action_spaces_dict={0: action_space},
        observation_spaces_dict={0: observation_space},
        agent_counts_dict={0: 1}
    )

    return rllib_model, maze_model_composer, observation_space, action_space


def reset_seed():
    """reset all seeds"""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.mark.rllib
def perform_test_maze_p_rllib_model_without_ray(gym_env: str):
    """Perfrom the test for a given gym env"""
    reset_seed()

    rllib_model, maze_model_composer, obs_space, act_space = build_default_cartpole_model(gym_env, MazeRLlibPolicyModel,
                                                                                          False)

    assert isinstance(rllib_model, nn.Module)
    assert isinstance(rllib_model, TorchModelV2)

    with pytest.raises(NotImplementedError):
        rllib_model.import_from_h5(None)

    test_obs_np = obs_space.sample()
    test_obs_torch = convert_to_torch(test_obs_np, device=None, cast=None, in_place=False)

    maze_logits = maze_model_composer.policy.networks[0](test_obs_torch)
    maze_logits_flat = torch.cat([maze_logits[action_keys] for action_keys in act_space.spaces.keys()], dim=-1)

    rllib_logits = rllib_model.forward({'obs': test_obs_torch}, state=[], seq_lens=torch.tensor([1], dtype=torch.int32))
    rllib_logits_flat = rllib_logits[0]

    assert torch.all(maze_logits_flat == rllib_logits_flat)

    with pytest.raises(NotImplementedError):
        rllib_model.value_function()

    try:
        import pygraphviz
        assert os.path.exists('policy_0.pdf')
    except ImportError:
        pass

    assert os.path.exists(SPACE_CONFIG_DUMP_FILE)
    os.remove(SPACE_CONFIG_DUMP_FILE)

    maze_state_dict = rllib_model.get_maze_state_dict(rllib_model.state_dict())
    assert 'policies' in maze_state_dict
    assert 0 in maze_state_dict['policies']
    assert isinstance(maze_state_dict['policies'][0], dict)
    for key, value in maze_state_dict['policies'][0].items():
        maze_model_state_dict = maze_model_composer.policy.networks[0].state_dict()
        assert key in maze_model_state_dict
        assert torch.allclose(maze_model_state_dict[key], value)


@pytest.mark.rllib
def test_maze_rllib_p_model_without_ray_cartpole():
    """Test cartpole model"""
    perform_test_maze_p_rllib_model_without_ray('CartPole-v0')


def perform_test_maze_ac_rllib_model_without_ray(gym_env: str):
    """Perfrom the test for a given gym env"""
    reset_seed()

    rllib_model, maze_model_composer, obs_space, act_space = build_default_cartpole_model(gym_env, MazeRLlibACModel,
                                                                                          True)
    assert isinstance(rllib_model, nn.Module)
    assert isinstance(rllib_model, TorchModelV2)

    with pytest.raises(NotImplementedError):
        rllib_model.import_from_h5(None)

    test_obs_np = obs_space.sample()
    test_obs_torch = convert_to_torch(test_obs_np, device=None, cast=None, in_place=False)

    maze_logits = maze_model_composer.policy.networks[0](test_obs_torch)
    maze_logits_flat = torch.cat([maze_logits[action_keys] for action_keys in act_space.spaces.keys()], dim=-1)

    rllib_logits = rllib_model.forward({'obs': test_obs_torch}, state=[], seq_lens=torch.tensor([1], dtype=torch.int32))
    rllib_logits_flat = rllib_logits[0]

    assert torch.all(maze_logits_flat == rllib_logits_flat)

    maze_value: torch.Tensor = maze_model_composer.critic.networks[0](test_obs_torch)['value']
    rllib_value = rllib_model.value_function()

    assert maze_value.equal(rllib_value)
    assert not torch.isnan(maze_value)

    try:
        import pygraphviz
        assert os.path.exists('critic_0.pdf')
        assert os.path.exists('policy_0.pdf')
    except ImportError:
        pass

    assert os.path.exists(SPACE_CONFIG_DUMP_FILE)
    os.remove(SPACE_CONFIG_DUMP_FILE)

    maze_state_dict = rllib_model.get_maze_state_dict(rllib_model.state_dict())
    assert 'policies' in maze_state_dict
    assert 0 in maze_state_dict['policies']
    assert isinstance(maze_state_dict['policies'][0], dict)
    for key, value in maze_state_dict['policies'][0].items():
        maze_model_state_dict = maze_model_composer.policy.networks[0].state_dict()
        assert key in maze_model_state_dict
        assert torch.allclose(maze_model_state_dict[key], value)

    assert 'critics' in maze_state_dict
    assert 0 in maze_state_dict['critics']
    assert isinstance(maze_state_dict['critics'][0], dict)
    for key, value in maze_state_dict['critics'][0].items():
        maze_model_state_dict = maze_model_composer.critic.networks[0].state_dict()
        assert key in maze_model_state_dict
        assert torch.allclose(maze_model_state_dict[key], value)


@pytest.mark.rllib
def test_maze_rllib_ac_model_without_ray_cartpole():
    """Test cartpole model"""
    perform_test_maze_ac_rllib_model_without_ray('CartPole-v0')


def perform_test_maze_q_rllib_model_without_ray(gym_env: str):
    """Perfrom the test for a given gym env"""
    reset_seed()

    rllib_model, maze_model_composer, obs_space, act_space = build_default_cartpole_model(gym_env, MazeRLlibQModel,
                                                                                          add_critic=True)

    assert isinstance(rllib_model, nn.Module)
    assert isinstance(rllib_model, TorchModelV2)
    assert isinstance(rllib_model, DQNTorchModel)

    with pytest.raises(NotImplementedError):
        rllib_model.import_from_h5(None)

    with pytest.raises(NotImplementedError):
        rllib_model.value_function()

    test_obs_np = obs_space.sample()
    test_obs_torch = convert_to_torch(test_obs_np, device=None, cast=None, in_place=False)

    maze_logits = maze_model_composer.policy.networks[0](test_obs_torch)
    maze_logits_flat = torch.cat([maze_logits[action_keys] for action_keys in act_space.spaces.keys()], dim=-1)

    test_obs_torch = convert_to_torch(test_obs_np, device=None, cast=None, in_place=False)
    rllib_logits = rllib_model.forward({'obs': test_obs_torch}, state=[], seq_lens=torch.tensor([1], dtype=torch.int32))
    rllib_logits_flat = rllib_logits[0]
    assert torch.allclose(maze_logits_flat, rllib_logits_flat)

    rllib_q_dist = rllib_model.get_q_value_distributions(rllib_logits_flat)
    assert torch.allclose(rllib_q_dist[0], rllib_logits_flat)

    logits_tmp = torch.unsqueeze(torch.ones_like(rllib_logits_flat), -1)
    assert torch.allclose(logits_tmp, rllib_q_dist[1])

    rllib_value = rllib_model.get_state_value(rllib_logits[0])
    assert not torch.isnan(rllib_value)

    assert os.path.exists('critic_0.pdf')
    assert os.path.exists('policy_0.pdf')

    os.remove('critic_0.pdf')
    os.remove('policy_0.pdf')

    assert os.path.exists(SPACE_CONFIG_DUMP_FILE)
    os.remove(SPACE_CONFIG_DUMP_FILE)

    maze_state_dict = rllib_model.get_maze_state_dict(rllib_model.state_dict())
    assert 'policies' in maze_state_dict
    assert 0 in maze_state_dict['policies']
    assert isinstance(maze_state_dict['policies'][0], dict)
    for key, value in maze_state_dict['policies'][0].items():
        maze_model_state_dict = maze_model_composer.policy.networks[0].state_dict()
        assert key in maze_model_state_dict
        assert torch.allclose(maze_model_state_dict[key], value)

    assert 'critic' in maze_state_dict
    assert 0 in maze_state_dict['critic']
    assert isinstance(maze_state_dict['critic'][0], dict)
    for key, value in maze_state_dict['critic'][0].items():
        maze_model_state_dict = maze_model_composer.critic.networks[0].state_dict()
        assert key in maze_model_state_dict
        assert torch.allclose(maze_model_state_dict[key], value)
