"""Composer for critic (value function) networks."""
from abc import abstractmethod
from typing import Dict, Union

from gym import spaces

from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.annotations import override
from maze.perception.models.critics.critic_composer_interface import CriticComposerInterface
from maze.perception.perception_utils import observation_spaces_to_in_shapes


class BaseStateCriticComposer(CriticComposerInterface):
    """Interface for critic (value function) network composers.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    """

    @abstractmethod
    def __init__(self, observation_spaces_dict: Dict[Union[str, int], spaces.Dict]):
        self._observation_spaces_dict = observation_spaces_dict

        # convert to observation shapes
        self._obs_shapes = observation_spaces_to_in_shapes(observation_spaces_dict)

    @property
    @abstractmethod
    @override(CriticComposerInterface)
    def critic(self) -> TorchStateCritic:
        """value networks"""
