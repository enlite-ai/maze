"""
A simple state observation which does nothing, except defining the observation space
"""

from typing import Dict

import gym
import numpy as np

from maze.core.env.observation_conversion import ObservationConversionInterface


class ObservationConversion(ObservationConversionInterface):
    """
    An state observation implementation
    """
    def maze_to_space(self, maze_state: Dict) -> Dict[str, np.ndarray]:
        """
        Does nothing
        :param maze_state: The state to pass through
        :return: The given state
        """
        return maze_state

    def space_to_maze(self, observation: Dict) -> Dict[str, np.ndarray]:
        """
        Does nothing
        :param observation: The observation to pass through
        :return: The given observation
        """
        return observation

    def space(self) -> gym.spaces.space.Space:
        """
        Important Note:
        This Dummy environment is programmed dynamically so you can just add observations starting with
        observation_0 -> observation 0 and observation 1
        or observation_1 -> only observation 1.

        :return: The finished gym observation space
        """
        return gym.spaces.Dict({
            "observation_0": gym.spaces.Box(shape=(3, 32, 32), low=0, high=1),
            "observation_1": gym.spaces.Box(shape=(10,), low=0, high=1),
        })
