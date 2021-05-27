"""Dummy structured policy for the CartPole env."""

from typing import Sequence, Tuple, Optional

import gym

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID


class DummyCartPolePolicy(Policy):
    """Dummy structured policy for the CartPole env.

    Useful mainly for showcase of the config scheme and for testing.
    """

    def __init__(self):
        self.action_space = gym.make("CartPole-v0").action_space

    def seed(self, seed: int) -> None:
        """Not applicable since heuristic is deterministic"""
        pass

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: ActorID = None,
                       deterministic: bool = False) -> ActionType:
        """Sample an action."""
        action = 1 if observation["observation"][2] > 0 else 0
        return {"action": action}

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: ActorID = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError
