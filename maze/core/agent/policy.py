"""Encapsulates policies and queries them for actions according to the provided policy ID."""

from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional

from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID


class Policy(ABC):
    """Structured policy class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates policies and queries them for actions according to the provided policy ID.
    """

    @abstractmethod
    def seed(self, seed: int) -> None:
        """Seed the policy to be used.
        Here the given seed should be used to initialize any random number generator or random state used to sample from
        distributions or any other form of randomness in the policy. This ensures that when the policy is explicit
        seeded it is reproducible.

        :param seed: The seed to use for all random state objects withing the policy.
        """

    @abstractmethod
    def needs_state(self) -> bool:
        """The policy implementation declares if it operates solely on observations (needs_state returns False) or
        if it also requires the state object in order to compute the action.

        Note that requiring the state object comes with performance implications, especially in multi-node distributed
        workloads, where both objects would need to be transferred over the network.
        """

    def needs_env(self) -> bool:
        """Similar to `needs_state`, the policy implementation declares if it operates solely on observations
        (needs_env returns False) or if it also requires the env object in order to compute the action.

        Requiring the env should be regarded as anti-pattern, but is supported for special cases like the MCTS policy,
        which requires cloning support from the environment.

        :return Per default policies return False.
        """
        return False

    @abstractmethod
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                       actor_id: Optional[ActorID] = None, deterministic: bool = False) -> ActionType:
        """
        Query a policy that corresponds to the given actor ID for action.

        :param observation: Current observation of the environment
        :param maze_state: Current state representation of the environment
                           (only provided if `needs_state()` returns True)
        :param env: The environment instance (only provided if `needs_env()` returns True)
        :param actor_id: ID of the actor to query policy for
                         (does not have to be provided if there is only one actor and one policy in this environment)
        :param deterministic: Specify if the action should be computed deterministically
        :return: Next action to take
        """

    @abstractmethod
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: Optional[int], maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: Optional[ActorID] = None, deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Get the top :num_candidates actions as well as the probabilities, q-values, .. leading to the decision.

        :param observation: Current observation of the environment
        :param num_candidates: The number of actions that should be returned. If None all candidates are returned.
        :param maze_state: Current state representation of the environment
                           (only provided if `needs_state()` returns True)
        :param env: The environment instance (only provided if `needs_env()` returns True)
        :param actor_id: ID of actor to query policy for
                         (does not have to be provided if policies dict contains only 1 policy)
        :param deterministic: Specify if the action should be computed deterministically
        :return: a tuple of sequences, where the first sequence corresponds to the possible actions, the other sequence
                 to the associated scores (e.g, probabilities or Q-values).
        """
