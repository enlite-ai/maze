"""Encapsulates state critic and queries them for values according to the provided policy ID."""

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, List

import torch
from maze.core.env.observation_conversion import ObservationType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


class StateCritic(ABC):
    """Structured state critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_values(self, record: StructuredSpacesRecord) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Query a critic that corresponds to the given ID for the state value.

        :param record: Record of a structured step containing keys and observations for the individual sub-steps
        :return: Tuple containing lists of values and detached values for individual sub-steps
        """

    @abstractmethod
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> Dict[str, torch.Tensor]:
        """Query a critic that corresponds to the given ID for the state value.

        :param observation: Current observation of the environment
        :param critic_id: The critic id to query
        :return: The value for this observation
        """
