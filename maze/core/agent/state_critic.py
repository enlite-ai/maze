"""Encapsulates state critic and queries them for values according to the provided policy ID."""

from __future__ import annotations

from abc import ABC, abstractmethod

from maze.core.agent.state_critic_input_output import (
    StateCriticInput,
    StateCriticOutput,
    StateCriticStepOutput,
)
from maze.core.env.observation_conversion import ObservationType


class StateCritic(ABC):
    """Structured state critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_values(self, critic_input: StateCriticInput) -> StateCriticOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param critic_input: The critic input for predicting the values of all sub-steps of the env.
        :return: Critic output holding the values, detached values and actor_id
        """

    @abstractmethod
    def predict_value(self, observation: ObservationType, critic_id: int | str) -> StateCriticStepOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param observation: Current observation of the environment
        :param critic_id: The critic id to query
        :return: The value for this observation
        """
