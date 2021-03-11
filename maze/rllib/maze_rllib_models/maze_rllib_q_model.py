"""This model holds an implementation of a Rllib DQN-Model compatible with maze."""
from typing import Tuple, List, Any, Dict, Optional

import gym.spaces as spaces
import torch
import torch.nn as nn
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.typing import TensorType

from maze.core.annotations import override, unused
from maze.core.utils.factory import ConfigType
from maze.rllib.maze_rllib_models.maze_rllib_base_model import MazeRLlibBaseModel


class MazeRLlibQModel(DQNTorchModel, MazeRLlibBaseModel):
    """Rllib Custom Model for q learning that works with the maze
    :class:`~maze.perception.models.model_composer.BaseModelComposer`.

    :param obs_space: Observation space of the target gym env. This object has an `original_space` attribute that
        specifies how to un-flatten the tensor into a ragged tensor.
    :param action_space: Action space of the target gym env
    :param num_outputs: Number of output units of the model
    :param model_config: config for the model, documented in ModelCatalog
    :param name: name (scope) for the model ConfigType dict.
    :param maze_model_composer_config: The config for the ModelComposer to be used. Arguments _action_spaces_dict, and
        _observation_spaces_dict are passed to the composer as instantiated objects (from the env).
    :param spaces_config_dump_file: Specify where the action/observation space config should be dumped.
    :param state_dict_dump_file: Specify where the state_dict should be dumped.
    :param dueling: Whether to build the advantage(A)/value(V) heads for DDQN. If True, Q-values are calculated as:
        Q = (A - mean[A]) + V. If False, raw NN output is interpreted.
        as Q-values.
    :param num_atoms: If >1, enables distributional DQN.
    :param kwargs: Other parameters given to the rllib qmodel that are never used in our implementation.
    """

    def __init__(self, obs_space: Any, action_space: spaces.Space, num_outputs: int, model_config: Dict, name: str,
                 maze_model_composer_config: ConfigType, spaces_config_dump_file: str, state_dict_dump_file: str,
                 dueling: bool = True, num_atoms: int = 1, **kwargs):
        unused(num_outputs)

        org_obs_space = obs_space.original_space

        assert isinstance(action_space, spaces.Discrete), f'Only discrete spaces supported but got {action_space}'

        num_outputs = action_space.n
        org_action_space = spaces.Dict({'action': action_space})

        assert dueling is True, 'Only dueling==True is supported at this point'
        assert num_atoms == 1, 'Only num_atoms == 1 is suported at this point'

        DQNTorchModel.__init__(self, obs_space=obs_space,
                               action_space=action_space,
                               model_config=model_config,
                               num_outputs=num_outputs,
                               name=name + '_maze_wrapper', dueling=dueling, num_atoms=num_atoms, **kwargs)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        MazeRLlibBaseModel.__init__(self, observation_space=org_obs_space, action_space=org_action_space,
                                    model_config=model_config, maze_model_composer_config=maze_model_composer_config,
                                    spaces_config_dump_file=spaces_config_dump_file,
                                    state_dict_dump_file=state_dict_dump_file)

        self._advantage_module: nn.Module = list(self.model_composer.policy.networks.values())[0]
        self.advantage_module = nn.Identity()

        # Assert that at most one network is used for critic
        assert self.model_composer.critic is not None and len(self.model_composer.critic.networks) == 1
        self._value_module: nn.Module = list(self.model_composer.critic.networks.values())[0]

        # Init class values
        self._value_module_input = None
        self._model_maze_input: Optional[Dict[str, torch.Tensor]] = None

    @override(DQNTorchModel)
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
        self._value_module_input = input_dict_maze

        return self.policy_forward(input_dict, self._advantage_module)

    @override(DQNTorchModel)
    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""
        assert self._value_module_input is not None, "must call forward() first"
        network_out = self._value_module(self._value_module_input)['value']
        return network_out

    @staticmethod
    @override(MazeRLlibBaseModel)
    def get_maze_state_dict(ray_state_dict: Dict):
        """implementation of
        :class:`~maze.rllib.custom_rllib.maze_rllib_models.maze_rllib_base_model.MazeRLlibBaseModel` interface
        """
        return MazeRLlibBaseModel._get_maze_state_dict(ray_state_dict, '_advantage_module.', '_value_module.')

    @override(DQNTorchModel)
    def import_from_h5(self, h5_file):
        """Import the model from h5"""
        raise NotImplementedError

    @override(DQNTorchModel)
    def value_function(self) -> TensorType:
        """Returns the value function output for the most recent forward pass.

        Note that a `forward` call has to be performed first, before this
        methods can return anything and thus that calling this method does not
        cause an extra forward pass through the network.

        :return: Value estimate tensor of shape [BATCH].
        """
        raise NotImplementedError('Value function not implemented for policy model')
