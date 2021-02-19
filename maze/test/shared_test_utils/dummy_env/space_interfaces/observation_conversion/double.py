"""Dummy state to observation interface. States and observations are represented as integers."""

from typing import Dict

import gym

from maze.core.env.observation_conversion import ObservationConversionInterface


class DoubleObservationConversion(ObservationConversionInterface):
    """
    Dummy action to MazeAction.

    Both states and observations are represented by integers. Observation equals double of the state.

    Note: Observation are wrapped in a dict to comply with the dict space requirement.
    """

    def space(self):
        """Numbers up to 1000 are allowed."""
        return gym.spaces.Dict({"observation": gym.spaces.Discrete(1000)})

    def maze_to_space(self, maze_state: int) -> Dict[str, int]:
        """Multiplies state by 2 and wraps it in a dict."""
        return {"observation": maze_state * 2}

    def space_to_maze(self, observation: Dict[str, int]) -> int:
        """Divides observation by 2."""
        observation = observation["observation"]
        assert observation % 2 == 0, "Invalid observation: Must be divisible by 2"
        return observation / 2
