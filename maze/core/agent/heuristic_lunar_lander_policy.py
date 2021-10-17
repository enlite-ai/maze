"""
Contains a simple heuristic policy for the OpenAI Gym LunarLander environment.

The implementation is adopted from here: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""

from typing import Union, Sequence, Tuple, Optional

import gym
import numpy as np

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID


class HeuristicLunarLanderPolicy(Policy):
    """Dummy structured policy for the LunarLander env.

    Useful mainly for testing the imitation learning pipeline.
    """

    def __init__(self):
        self.action_space = gym.make("LunarLander-v2").action_space

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Not applicable since heuristic is deterministic"""
        pass

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: ActorID = None,
                       deterministic: bool = False) -> ActionType:
        """Sample an action."""

        s = observation["observation"]

        # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        angle_targ = s[0] * 0.5 + s[2] * 1.0
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(s[0])  # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
        # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1

        return {"action": a}

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: Union[str, int] = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError
