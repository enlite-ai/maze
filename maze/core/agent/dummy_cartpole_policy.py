"""Dummy structured policy for the CartPole env."""

from typing import Union, Sequence, Tuple, Optional

import gym
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType


class DummyCartPolePolicy(Policy):
    """Dummy structured policy for the CartPole env.

    Useful mainly for showcase of the config scheme and for testing.
    """

    def __init__(self):
        self.action_space = gym.make("CartPole-v0").action_space

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType] = None,
                       policy_id: Union[str, int] = None, deterministic: bool = False) -> ActionType:
        """Sample an action."""
        action = 1 if observation["observation"][2] > 0 else 0
        return {"action": action}

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType] = None,
                                      policy_id: Union[str, int] = None, deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError
