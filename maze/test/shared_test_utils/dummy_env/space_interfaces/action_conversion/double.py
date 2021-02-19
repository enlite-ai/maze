"""Dummy action conversion. Actions and MazeActions are represented as integers."""
from typing import Dict

import gym

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_state import MazeStateType


class DoubleActionConversion(ActionConversionInterface):
    """
    Dummy action conversion interface implementation.

    Both MazeActions and actions are represented by integers. Action equals double of the MazeAction.

    Note: Actions are wrapped in a dict to comply with the dict space requirement.
    """

    def space(self):
        """Numbers up to 1000 are allowed."""
        return gym.spaces.Dict({"action": gym.spaces.Discrete(1000)})

    def space_to_maze(self, action: Dict[str, int], maze_state: MazeStateType) -> int:
        """Divides action by 2."""
        action = action["action"]
        assert action % 2 == 0, "Invalid action: Must be divisible by 2"
        return action / 2

    def maze_to_space(self, maze_action: int) -> Dict[str, int]:
        """Multiplies MazeAction by two and wraps it in a dict."""
        return {"action": maze_action * 2}
