"""Composer for policy (actor) networks."""
from abc import ABC, abstractmethod
from typing import Dict, Union

from gym import spaces

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StepKeyType
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import observation_spaces_to_in_shapes


class BasePolicyComposer(ABC):
    """Interface for policy (actor) network composers.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper: The distribution mapper.
    """

    def __init__(self,
                 action_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 distribution_mapper: DistributionMapper):
        self._action_spaces_dict = action_spaces_dict
        self._observation_spaces_dict = observation_spaces_dict
        self._agent_counts_dict = agent_counts_dict
        self._distribution_mapper = distribution_mapper

        # convert to observation shapes
        self._obs_shapes = observation_spaces_to_in_shapes(observation_spaces_dict)

        # convert to action shapes
        self._action_logit_shapes = \
            {step_key: {action_head: self._distribution_mapper.required_logits_shape(action_head)
                        for action_head in action_spaces_dict[step_key].spaces.keys()}
             for step_key in action_spaces_dict.keys()}

    @property
    @abstractmethod
    def policy(self) -> TorchPolicy:
        """The policy object"""
