"""
Includes the implementation of the dummy core environment.
"""
from typing import Tuple, Dict, Any, Union, Optional

import gym
import numpy as np

from maze.core.env.core_env import CoreEnv
from maze.core.rendering.renderer import Renderer
from maze.core.utils.seeding import set_random_states
from maze.test.shared_test_utils.dummy_env.reward.base import RewardAggregator


class DummyCoreEnvironment(CoreEnv):
    """
    Does as little as possible, returns random actions

    :param observation_space: The observation space for the environment (in the state to observation interface)
    """

    def __init__(self, observation_space: gym.spaces.space.Space):
        super().__init__()

        self.reward_aggregator = RewardAggregator()
        self.observation_space = observation_space

    def step(self, maze_action: Dict) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict]]:
        """
        :param maze_action: Environment MazeAction to take.
        :return: state, reward, done, info
        """

        return self.get_maze_state(), 10, False, {}

    def get_maze_state(self) -> Dict[str, np.ndarray]:
        """
        :returns Random observation
        """
        return self.observation_space.sample()

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Does nothing
        :return: The environment state
        """
        return self.get_maze_state()

    def render(self, mode='human'):
        """
        Not implemented
        """
        pass

    def close(self):
        """
        Not implemented
        """
        pass

    def seed(self, seed: int):
        """
        Sets the seed for the environment
        :param seed: The given seed
        """
        set_random_states(seed)

    def get_serializable_components(self) -> Dict[str, Any]:
        """
        Not implemented
        :return: An empty dict
        """
        return {}

    def get_renderer(self) -> Optional[Renderer]:
        """
        Not implemented
        :return: None
        """
        return None

    def actor_id(self) -> Tuple[Union[str, int], int]:
        """
        Not implemented
        :return: 0, 0
        """
        return 0, 0

    def is_actor_done(self) -> bool:
        """
        Not implemented
        :return: False
        """
        return False
