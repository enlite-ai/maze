"""Encapsulates state critic and queries them for values according to the provided policy ID."""

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple

import torch
from maze.core.env.observation_conversion import ObservationType


class StateCritic(ABC):
    """Structured state critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_values(self, observations: Dict[Union[str, int], ObservationType]) -> \
            Tuple[Dict[Union[str, int], torch.Tensor],
                  Dict[Union[str, int], torch.Tensor]]:
        """Query a critic that corresponds to the given ID for the state value.

        :param observations: Current observation of the environment
        :return: Tuple containing the values and detached values
        """

    @abstractmethod
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> torch.Tensor:
        """Query a critic that corresponds to the given ID for the state value.

        :param observation: Current observation of the environment
        :param critic_id: The critic id to query
        :return: The value for this observation
        """
