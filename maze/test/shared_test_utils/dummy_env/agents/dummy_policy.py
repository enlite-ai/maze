"""
Implementation of a dummy policy for the DummyEnvironment.
"""
from typing import Sequence, Tuple, Union, Optional

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID


class DummyGreedyPolicy(Policy):
    """
    Dummy greedy policy for DummyEnvironment. Recommended action is constant w.r.t. the specified observation.
    """

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Not applicable since heuristic is deterministic"""
        pass

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: Optional[ActorID] = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Not implemented.
        """
        raise NotImplementedError

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """
        Returns next action to take.
        :return: Action derived from observation state. Constant w.r.t. specified observation.
        """

        # Derive base for action value from specified observation.
        val: float = sum([np.sum(observation[key]) for key in observation])

        return {
            "action_0_0": int(val % 10),
            "action_1_0": int(val % 10),
            "action_1_1": np.asarray([round(val - int(val))] * 5, dtype=np.int8)
        }
