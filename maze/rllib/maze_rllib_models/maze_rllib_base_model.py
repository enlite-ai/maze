"""This file contains the rllib compatible maze Model"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict, Optional

import gym.spaces as spaces
import torch
import torch.nn as nn

from maze.core.annotations import unused
from maze.core.utils.factory import ConfigType, Factory
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.space_config import SpacesConfig
from maze.perception.perception_utils import convert_to_torch


class MazeRLlibBaseModel(ABC):
    """Rllib Custom Model that works with the maze :class:`~maze.perception.models.model_composer.BaseModelComposer`.

    :param observation_space: Observation space of the target gym env. This object has an `original_space` attribute
        that specifies how to un-flatten the tensor into a ragged tensor.
    :param action_space: Action space of the target gym env.
    :param model_config: config for the model, documented in ModelCatalog
        ConfigType dict (to be created thought the registry).
    :param maze_model_composer_config: The config for the ModelComposer to be used. Arguments _action_spaces_dict, and
        _observation_spaces_dict are passed to the composer as instantiated objects (from the env).
    :param spaces_config_dump_file: Specify where the action/observation space config should be dumped.
    :param state_dict_dump_file: Specify the name of the state_dict_dump_file.
    """

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Dict, model_config: Dict,
                 maze_model_composer_config: ConfigType, spaces_config_dump_file: str, state_dict_dump_file: str):

        unused(state_dict_dump_file)

        assert isinstance(action_space, spaces.Dict), f'The given action_space should be gym.spaces.Dict, but we' \
                                                      f' got {type(action_space)}'

        assert isinstance(observation_space, spaces.Dict), f'The original observation space has to be a gym Dict, ' \
                                                           f'but we got {type(observation_space)}'

        assert model_config.get('vf_share_layers') is False, 'vf_share_layer not implemented for maze models'

        # Initialize model composer
        self.model_composer = Factory(BaseModelComposer).instantiate(
            maze_model_composer_config,
            action_spaces_dict={0: action_space},
            observation_spaces_dict={0: observation_space},
            agent_counts_dict={0: 1}
        )

        # Obtain action order from distribution mapper (this has to be in the same order as the attribute
        #   self.action_heads in the MazeRLlibActionDistribution
        self.action_keys = list(self.model_composer.distribution_mapper.action_space.spaces.keys())

        # Initialize space config, and dump it to file
        SpacesConfig(self.model_composer.action_spaces_dict,
                     self.model_composer.observation_spaces_dict,
                     self.model_composer.agent_counts_dict).save(spaces_config_dump_file)

        # Assert that only one network is used for policy
        assert len(self.model_composer.policy.networks) == 1

    def policy_forward(self, input_dict: Dict[str, Any], policy_model: nn.Module) -> Tuple[Any, List]:
        """Perform the forward pass through the network

        :param input_dict: Dictionary of input tensors, including "obs",
            "obs_flat", "prev_action", "prev_reward", "is_training"
        :param policy_model: The policy used to compute the actions
        :return: A tuple of network output and state, where the network output tensor is of
            size [BATCH, num_outputs]
        """

        input_dict_maze = input_dict['obs']
        assert isinstance(input_dict_maze, dict), f'input_dict[\'obs\'] has to be a dictionary itself, but we got' \
                                                  f' {type(input_dict_maze)}'

        network_out = policy_model(input_dict_maze)
        network_out_flat = torch.cat([network_out[action_keys] for action_keys in self.action_keys], dim=-1)

        return network_out_flat, []

    @staticmethod
    @abstractmethod
    def get_maze_state_dict(ray_state_dict: Dict) -> Dict:
        """Get the maze compatible version of the state_dict from the ray_state_dict.

        :param ray_state_dict: The state dict constructed by calling the state_dict method of this whole model.
        :return: The maze compatible state dict.
        """

    @staticmethod
    def _get_maze_state_dict(ray_state_dict: Dict, policy_att_name: str, critic_att_name: Optional[str]) -> Dict:
        """Get the maze compatible version of the state_dict from the ray_state_dict.

        :param ray_state_dict: The state dict constructed by calling the state_dict method of this whole model.
        :param policy_att_name: The attribute name of the policy
        :param critic_att_name: The attribute name of the critic
        :return: The maze compatible state dict.
        """

        # collect state dict for models and optimizer
        state_dict = dict()
        state_dict["policies"] = {0: {}}
        if critic_att_name is not None:
            state_dict["critics"] = {0: {}}

        for key, weight in ray_state_dict.items():
            if key.startswith(policy_att_name):
                state_dict['policies'][0][key.replace(policy_att_name, '')] = convert_to_torch(weight, cast=None,
                                                                                               device='cpu',
                                                                                               in_place=False)
            elif critic_att_name is not None and key.startswith(critic_att_name):
                state_dict['critics'][0][key.replace(critic_att_name, '')] = convert_to_torch(weight, cast=None,
                                                                                              device='cpu',
                                                                                              in_place=False)
        assert len(state_dict['policies'][0]) > 0

        return state_dict
