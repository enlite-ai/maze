"""Encapsulates state critic and queries them for values according to the provided policy ID."""
import dataclasses
from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, List

import torch
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StepKeyType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord

@dataclasses.dataclass
class CriticSubstepOutput:
    values: torch.Tensor

    detached_values: torch.Tensor

class CriticOutput:
    def __init__(self):
        self._step_critic_outputs: Dict[StepKeyType, CriticSubstepOutput] = dict()

    def __setitem__(self, key: StepKeyType, value: CriticSubstepOutput):
        """ Set self[key] to value. """
        self._step_critic_outputs[key] = value

    def keys(self):
        return self._step_critic_outputs.keys()

    @property
    def values(self):
        return {step_key: self._step_critic_outputs[step_key].values for step_key in self.keys()}

    @property
    def detached_values(self):
        return {step_key: self._step_critic_outputs[step_key].detached_values for step_key in self.keys()}

    def reshape(self, shape):
        for step_idx, value in self._step_critic_outputs.items():
            self._step_critic_outputs[step_idx].values = value.values.reshape(shape)
            self._step_critic_outputs[step_idx].detached_values = value.detached_values.reshape(shape)


CriticInput = Dict[StepKeyType, torch.Tensor]


class StateCritic(ABC):
    """Structured state critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_values(self, record: CriticInput) -> CriticOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param record: TODO:
        :return: Tuple containing lists of values and detached values for individual sub-steps
        """

    @abstractmethod
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> CriticSubstepOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param observation: Current observation of the environment
        :param critic_id: The critic id to query
        :return: The value for this observation
        """
