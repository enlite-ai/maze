"""Loads trained policies for rollout in structured environments."""
from typing import Union, Dict, Optional

import torch
from omegaconf import DictConfig

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.factory import Factory
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.space_config import SpacesConfig


class SerializedTorchPolicy(TorchPolicy):
    """Structured policy used for rollouts of trained models.

    Will build the models based on the model composer and spaces config and set the state of individual
    policies according to the state dict dump.

    Policies are set to eval mode by default.

    :param model: Model composer configuration
    :param state_dict_file: Path to dumped state dictionaries of the trained policies
    :param spaces_dict_file: Path to dumped spaces configuration (action and observation spaces of
                                  the env the policy was trained on, used for model initialization)
    :param deterministic: If True actions are computed deterministically; else sample from the probability distribution.
    """

    def __init__(self,
                 model: Union[DictConfig, Dict],
                 state_dict_file: str,
                 spaces_dict_file: str,
                 device: str,
                 deterministic: bool):
        spaces_config = SpacesConfig.load(spaces_dict_file)
        model_composer = Factory(base_type=BaseModelComposer).instantiate(
            model,
            action_spaces_dict=spaces_config.action_spaces_dict,
            observation_spaces_dict=spaces_config.observation_spaces_dict,
            agent_counts_dict=spaces_config.agent_counts_dict
        )

        super().__init__(networks=model_composer.policy.networks,
                         distribution_mapper=model_composer.distribution_mapper,
                         device=device,
                         substeps_with_separate_agent_nets=model_composer.policy.substeps_with_separate_agent_nets)

        state_dict = torch.load(state_dict_file, map_location=torch.device(self._device))

        self.load_state_dict(state_dict)
        self.eval()

        self.deterministic = deterministic

    @override(TorchPolicy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: ActorID = None,
                       deterministic: bool = False) -> ActionType:
        return super().compute_action(observation=observation, maze_state=maze_state, env=env, actor_id=actor_id,
                                      deterministic=self.deterministic)

    @override(TorchPolicy)
    def seed(self, seed: int):
        """Set torch manual seed"""
        torch.manual_seed(seed)
