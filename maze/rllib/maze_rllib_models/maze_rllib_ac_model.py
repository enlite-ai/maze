"""This file contains the rllib compatible actor-critic maze Model"""
from typing import Tuple, List, Any, Dict, Optional

import gym.spaces as spaces
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from maze.core.annotations import override
from maze.core.utils.factory import ConfigType
from maze.rllib.maze_rllib_models.maze_rllib_policy_model import MazeRLlibPolicyModel


class MazeRLlibACModel(MazeRLlibPolicyModel):
    """Rllib Custom Actor Critic Model that works with the maze
    :class:`~maze.perception.models.model_composer.BaseModelComposer`.

    :param obs_space: Observation space of the target gym env. This object has an `original_space` attribute that
        specifies how to un-flatten the tensor into a ragged tensor.
    :param action_space: Action space of the target gym env
    :param num_outputs: Number of output units of the model
    :param model_config: config for the model, documented in ModelCatalog
    :param name: name (scope) for the model
        ConfigType dict (to be created thought the registry)
    :param maze_model_composer_config: The config for the ModelComposer to be used. Arguments _action_spaces_dict, and
        _observation_spaces_dict are passed to the composer as instantiated objects (from the env).
    :param spaces_config_dump_file: Specify where the action/observation space config should be dumped.
    :param state_dict_dump_file: Specify where the state_dict should be dumped.
    """

    def __init__(self, obs_space: Any, action_space: spaces.Dict, num_outputs: int, model_config: Dict, name: str,
                 maze_model_composer_config: ConfigType, spaces_config_dump_file: str, state_dict_dump_file: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, maze_model_composer_config,
                         spaces_config_dump_file, state_dict_dump_file)

        # Assert that at most one network is used for critic
        assert self.model_composer.critic is not None and len(self.model_composer.critic.networks) == 1, \
            'Please specify a critic when using actor critic algorithms'
        self._critic: nn.Module = list(self.model_composer.critic.networks.values())[0]

        # Init class values
        self._cur_value: Optional[torch.Tensor] = None
        self._critic_input: Optional[Dict[str, torch.Tensor]] = None

    @override(MazeRLlibPolicyModel)
    def forward(self, input_dict: Dict[str, Any], state: List, seq_lens: torch.Tensor) -> Tuple[Any, List]:
        """Perform the forward pass through the network

        :param input_dict: Dictionary of input tensors, including "obs",
            "obs_flat", "prev_action", "prev_reward", "is_training"
        :param state: List of state tensors with sizes matching those
            returned by get_initial_state + the batch dimension
        :param seq_lens: 1d tensor holding input sequence lengths
        :return: A tuple of network output and state, where the network output tensor is of
            size [BATCH, num_outputs]
        """

        input_dict_maze = input_dict['obs']
        self._critic_input = input_dict_maze

        return super().forward(input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self) -> float:
        """Returns the value function output for the most recent forward pass.

        Note that a `forward` call has to be performed first, before this
        methods can return anything and thus that calling this method does not
        cause an extra forward pass through the network.

        :return: Value estimate tensor of shape [BATCH].
        """
        assert self._critic_input is not None, "must call forward() first"
        out_value = self._critic(self._critic_input)['value']
        if len(out_value.shape) > 1:
            out_value = out_value.squeeze(1)
        self._cur_value = out_value
        return self._cur_value

    @staticmethod
    @override(MazeRLlibPolicyModel)
    def get_maze_state_dict(ray_state_dict: Dict):
        """implementation of
        :class:`~maze.rllib.custom_rllib.maze_rllib_models.maze_rllib_base_model.MazeRLlibBaseModel` interface
        """
        return MazeRLlibPolicyModel._get_maze_state_dict(ray_state_dict, '_policy.', '_critic.')
