"""Loads a TorchActorCritic instance with its structured policy and critic from disk."""
from typing import Union, Dict

import torch
from omegaconf import DictConfig

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.utils.factory import Factory
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.space_config import SpacesConfig


class SerializedActorCritic(TorchActorCritic):
    """TorchActorCritic with structured policy and critic, loaded from disk.

    Will build the models based on the model composer and spaces config and set the state of individual
    policies according to the state dict dump.

    Policies are set to eval mode by default.

    :param model: Model composer configuration
    :param state_dict_file: Path to dumped state dictionaries of the trained policies
    :param spaces_dict_file: Path to dumped spaces configuration (action and observation spaces of
                                  the env the policy was trained on, used for model initialization)
    """

    def __init__(self,
                 model: Union[DictConfig, Dict],
                 state_dict_file: str, spaces_dict_file: str, device: str):
        spaces_config = SpacesConfig.load(spaces_dict_file)
        model_composer = Factory(base_type=BaseModelComposer).instantiate(
            model,
            action_spaces_dict=spaces_config.action_spaces_dict,
            observation_spaces_dict=spaces_config.observation_spaces_dict
        )

        super().__init__(policy=model_composer.policy,
                         critic=model_composer.critic,
                         device=device)

        state_dict = torch.load(state_dict_file, map_location=torch.device(self._device))

        self.load_state_dict(state_dict)
        self.eval()
