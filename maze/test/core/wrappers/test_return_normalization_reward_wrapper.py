"""Contains reward normalization unit tests."""
from maze.core.wrappers.return_normalization_reward_wrapper import ReturnNormalizationRewardWrapper
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

    env = ReturnNormalizationRewardWrapper(env, gamma=0.99, epsilon=1e-8)
    env.reset()
    reward = env.step(env.action_space.sample())[1]
    assert isinstance(reward, float)
    assert not hasattr(reward, 'shape')
