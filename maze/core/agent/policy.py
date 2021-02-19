"""Encapsulates policies and queries them for actions according to the provided policy ID."""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Sequence, Optional

from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType


class Policy(ABC):
    """Structured policy class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates policies and queries them for actions according to the provided policy ID.
    """

    @abstractmethod
    def needs_state(self) -> bool:
        """The policy implementation declares if it operates solely on observations (needs_state returns False) or
        if it also requires the state object in order to compute the action.

        Note that requiring the state object comes with performance implications, especially in multi-node distributed
        workloads, where both objects would need to be transferred over the network.
        """

    @abstractmethod
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType],
                       policy_id: Union[str, int] = None, deterministic: bool = False) -> ActionType:
        """
        Query a policy that corresponds to the given ID for action.

        :param observation: Current observation of the environment
        :param maze_state: Current state representation of the environment (only provided if `needs_state()` returns True)
        :param policy_id: ID of the policy to query
                          (does not have to be provided if policies dict contains only 1 policy)
        :param deterministic: Specify if the action should be computed deterministically
        :return: Next action to take
        """

    @abstractmethod
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType],
                                      policy_id: Union[str, int] = None, deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Get the top :num_candidates actions as well as the probabilities, q-values, .. leading to the decision.

        :param observation: Current observation of the environment
        :param num_candidates: The number of actions that should be returned
        :param maze_state: Current state representation of the environment (only provided if `needs_state()` returns True)
        :param policy_id: ID of the policy to query
                          (does not have to be provided if policies dict contains only 1 policy)
        :param deterministic: Specify if the action should be computed deterministically
        :return: a tuple of sequences, where the first sequence corresponds to the possible actions, the other sequence
                 to the associated scores (e.g, probabilities or Q-values).
        """
