"""
A simple action conversion which does nothing, except defining the action space
"""

from typing import Dict, Any

import gym

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_state import MazeStateType


class DictActionConversion(ActionConversionInterface):
    """
    An action conversion interface implementation
    """
    def space_to_maze(self, action: Dict, maze_state: MazeStateType) -> Dict[str, Any]:
        """
        Does nothing
        :param action: The action to pass through
        :param maze_state: Not used in this case
        :return: The action
        """
        return action

    def space(self) -> gym.spaces.space.Space:
        """
        Important Note:
        This Dummy environment is programmed dynamically so you can just add actions starting with action_0 or action_1.

        :return: The finished gym action space
        """
        return gym.spaces.Dict({
            "action_0_0": gym.spaces.Discrete(10),
            "action_0_1": gym.spaces.MultiDiscrete([3, 5]),
            "action_0_2": gym.spaces.Box(low=-1, high=1, shape=(5,)),
            "action_1_0": gym.spaces.Discrete(10),
            "action_1_1": gym.spaces.MultiBinary(5),
            "action_2_0": gym.spaces.Box(low=-5, high=5, shape=(5,)),
        })

    def noop_action(self):
        """Converts environment MazeAction to agent action.

        :return: the noop action.
        """
        return self.space().sample()
