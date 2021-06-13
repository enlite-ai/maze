"""Contains utility functions for pre-processor tests."""
import gym
import numpy as np

from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


class PreProcessingObservationConversion(ObservationConversion):
    """
    An observation conversion implementation
    """

    def space(self) -> gym.spaces.space.Space:
        """
        Important Note:
        This Dummy environment is programmed dynamically so you can just add observations starting with
        observation_0 -> observation 0 and observation 1
        or observation_1 -> only observation 1.

        :return: The finished gym observation space
        """
        return gym.spaces.Dict({
            "observation_0_feature_series":
                gym.spaces.Box(low=np.float32(0), high=np.float32(1), shape=(64, 24), dtype=np.float32),
            "observation_0_image":
                gym.spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "observation_1_categorical_feature":
                gym.spaces.Box(low=np.float32(0), high=np.float32(11), shape=(), dtype=np.float32),
        })


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """
    Instantiates the DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """

    observation_conversion = PreProcessingObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )

    return DummyStructuredEnvironment(maze_env=maze_env)
