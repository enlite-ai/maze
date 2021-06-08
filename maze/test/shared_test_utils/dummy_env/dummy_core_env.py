"""
Includes the implementation of the dummy core environment.
"""
from typing import Tuple, Dict, Any, Optional

import gym
import numpy as np

from maze.core.env.core_env import CoreEnv
from maze.core.env.structured_env import StepKeyType, ActorID
from maze.core.events.pubsub import Pubsub
from maze.core.rendering.renderer import Renderer
from maze.test.shared_test_utils.dummy_env.dummy_renderer import DummyMatplotlibRenderer
from maze.test.shared_test_utils.dummy_env.reward.base import RewardAggregator, DummyEnvEvents


class DummyCoreEnvironment(CoreEnv):
    """
    Does as little as possible, returns random actions

    :param observation_space: The observation space for the environment (in the state to observation interface)
    """

    def __init__(self, observation_space: gym.spaces.space.Space):
        super().__init__()

        self.pubsub = Pubsub(self.context.event_service)
        self.dummy_core_events = self.pubsub.create_event_topic(DummyEnvEvents)

        self.reward_aggregator = RewardAggregator()
        self.pubsub.register_subscriber(self.reward_aggregator)

        self.observation_space = observation_space

        # initialize rendering
        self.renderer = DummyMatplotlibRenderer()

    def step(self, maze_action: Dict) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict]]:
        """
        :param maze_action: Environment MazeAction to take.
        :return: state, reward, done, info
        """

        self.dummy_core_events.twice_per_step(3)
        self.dummy_core_events.twice_per_step(7)

        return self.get_maze_state(), self.reward_aggregator.summarize_reward(), False, {}

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
        # No randomness in the env
        pass

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
        return self.renderer

    def actor_id(self) -> ActorID:
        """Single-step, single-agent environment"""
        return ActorID(step_key=0, agent_id=0)

    def is_actor_done(self) -> bool:
        """
        Not implemented
        :return: False
        """
        return False

    @property
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Single-step, single agent env."""
        return {0: 1}
