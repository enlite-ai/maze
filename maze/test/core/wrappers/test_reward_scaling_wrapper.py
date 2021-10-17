"""Contains reward scaling unit tests."""
import numpy as np

from maze.core.wrappers.reward_scaling_wrapper import RewardScalingWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def test_reward_scaling_wrapper():
    """ Unit tests """
    observation_conversion = ObservationConversion()

    env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion])

    action = env.action_space.sample()
    np.random.seed(1234)
    original_reward = env.step(action)[1]

    wrapped_env = RewardScalingWrapper(env, scale=0.1)
    np.random.seed(1234)
    wrapped_reward = wrapped_env.step(action)[1]

    assert original_reward == wrapped_reward * 10
