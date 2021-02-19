""" Contains helper functions for unit testing """
from typing import Tuple, Any, Dict

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def build_dummy_base_env() -> BaseEnv:
    """ helper function creating a DummyBaseEnv for unit testing.
    :return: A Dummy Base Env.
    """

    class SomeBaseEnv(BaseEnv):
        """A dummy base env for unit testing.
        """

        @override(BaseEnv)
        def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
            """ override of BaseEnv """
            return None, None, False, {}

        @override(BaseEnv)
        def reset(self) -> Any:
            """ override of BaseEnv """
            return None

        @override(BaseEnv)
        def seed(self, seed: int) -> None:
            """ override of BaseEnv """
            pass

        @override(BaseEnv)
        def close(self) -> None:
            """ override of BaseEnv """
            pass

    return SomeBaseEnv()


def build_dummy_maze_env() -> DummyEnvironment:
    """
    Instantiates the DummyEnvironment.

    :return: Instance of a DummyEnvironment
    """
    observation_conversion = ObservationConversion()

    return DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )


def build_dummy_structured_env() -> DummyStructuredEnvironment:
    """
    Instantiates a DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """
    observation_conversion = ObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )

    return DummyStructuredEnvironment(maze_env=maze_env)
