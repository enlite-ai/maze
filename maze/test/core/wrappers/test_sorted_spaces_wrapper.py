"""Contains sorted spaces wrapper unit tests."""
from typing import Dict, Any

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.wrappers.sorted_spaces_wrapper import SortedSpacesWrapper
from maze.core.wrappers.wrapper import ActionWrapper, EnvType, ObservationWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


class AddActionWrapper(ActionWrapper[EnvType]):
    """Test add action wrapper"""

    def __init__(self, env: EnvType):
        super().__init__(env)
        action_spaces_dict = self.env.action_space.spaces
        action_spaces_dict['000_action'] = gym.spaces.Discrete(10)

        self.action_space = gym.spaces.Dict(action_spaces_dict)

    def action(self, action: Any) -> Any:
        """Abstract action mapping method."""
        action['000_action'] = 6
        return action

    def reverse_action(self, action: Any) -> Any:
        """Abstract action reverse mapping method."""
        raise NotImplementedError


class AddObservationWrapper(ObservationWrapper[EnvType]):
    """Test add observation wrapper"""

    def __init__(self, env: EnvType):
        super().__init__(env)
        self.new_shape = (3,)
        observation_spaces_dict = self.env.observation_space.spaces
        observation_spaces_dict['0000_obs'] = gym.spaces.Box(low=np.float32(0), high=np.float32(1),
                                                             shape=self.new_shape, dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_spaces_dict)

    @override(ObservationWrapper)
    def observation(self, observation) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ObservationWrapper` interface.
        """
        observation['0000_obs'] = np.random.random(self.new_shape)
        return observation


def test_sorted_action_spaces_wrapper():
    """ Unit tests """
    observation_conversion = ObservationConversion()

    env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion])
    env = AddActionWrapper.wrap(env)

    env = SortedSpacesWrapper.wrap(env)

    sorted_action_space = env.action_space
    assert list(sorted_action_space.spaces.keys()) == list(sorted(sorted_action_space.spaces.keys()))


def test_sorted_observation_spaces_wrapper():
    """ Unit tests """
    observation_conversion = ObservationConversion()

    env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion])
    env = AddObservationWrapper.wrap(env)

    env = SortedSpacesWrapper.wrap(env)

    sorted_obs_space = env.observation_space
    assert list(sorted_obs_space.spaces.keys()) == list(sorted(sorted_obs_space.spaces.keys()))

    _, _ = env.get_observation_and_action_dicts({}, env.last_maze_action, first_step_in_episode=True)
