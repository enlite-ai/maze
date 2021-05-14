"""Core Interface for implementing a custom policy in a given env."""

from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional

from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType


class FlatPolicy(ABC):
    """Generic flat policy interface."""

    @abstractmethod
    def compute_action(self, observation: ObservationType, deterministic: bool) -> ActionType:
        """
        Pick the next action based on the current observation.

        :param observation: Current observation of the environment
        :param deterministic: Specify if the action should be computed deterministically
        :return: Next action to take
        """

    @abstractmethod
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int]) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Get the top :num_candidates actions as well as the probabilities, q-values, .. leading to the decision.

        :param observation: Current observation of the environment
        :param num_candidates: The number of actions that should be returned
        :return: a tuple of sequences, where the first sequence corresponds to the possible actions, the other sequence
                 to the associated probabilities
        """
