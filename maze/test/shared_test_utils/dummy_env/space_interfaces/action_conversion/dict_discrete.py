"""
A simple action conversion which does nothing, except defining the action space as a dict of several
 discrete action spaces
"""

from typing import Dict, Any

import gym

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_state import MazeStateType


class DictDiscreteActionConversion(ActionConversionInterface):
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
            "action_0_0": gym.spaces.Discrete(10)
        })

    def noop_action(self):
        """Converts environment MazeAction to agent action.

        :return: the noop action.
        """
        t = self.space().sample()
        t['action_0_0'] = 0
        return t
