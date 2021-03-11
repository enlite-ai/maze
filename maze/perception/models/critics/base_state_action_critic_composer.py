"""Composer for state action (Q) critic networks."""
from abc import abstractmethod
from typing import Dict, Union

from gym import spaces

from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.annotations import override
from maze.perception.models.critics.critic_composer_interface import CriticComposerInterface
from maze.perception.perception_utils import observation_spaces_to_in_shapes


class BaseStateActionCriticComposer(CriticComposerInterface):
    """Interface for state action (Q) critic network composers.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param action_spaces_dict: Dict of sub-step id to action space.
    """

    @abstractmethod
    def __init__(self, observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 action_spaces_dict: Dict[Union[str, int], spaces.Dict]):
        self._observation_spaces_dict = observation_spaces_dict
        self._action_spaces_dict = action_spaces_dict
        # convert to observation shapes
        self._obs_shapes = observation_spaces_to_in_shapes(observation_spaces_dict)

        # Check whether only discrete spaces are present in each step!!!
        self._only_discrete_spaces = {step_key: True for step_key in self._obs_shapes.keys()}
        for step_key, dict_action_space in self._action_spaces_dict.items():
            for act_key, act_space in dict_action_space.spaces.items():
                assert isinstance(act_space, (spaces.Discrete, spaces.Box)), 'Only box and discrete spaces supported ' \
                                                                             'thus far'
                if self._only_discrete_spaces[step_key] and not isinstance(act_space, spaces.Discrete):
                    self._only_discrete_spaces[step_key] = False

    @property
    @abstractmethod
    @override(CriticComposerInterface)
    def critic(self) -> TorchStateActionCritic:
        """value networks"""
