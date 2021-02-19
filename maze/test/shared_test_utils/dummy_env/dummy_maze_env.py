"""
Includes the implementation of the dummy flat environment.
"""

from typing import Union

import maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict as action_conversion_module
import maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion as observation_conversion_module
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.registry import Registry, CollectionOfConfigType
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment


class DummyEnvironment(MazeEnv):
    """
    Environment.
    """
    # Dynamically register action conversion interfaces
    action_conversion_registry = Registry(base_type=ActionConversionInterface,
                                          root_module=action_conversion_module)

    # Dynamically register observation conversion interfaces
    observation_conversion_registry = Registry(base_type=ObservationConversionInterface,
                                               root_module=observation_conversion_module)

    def __init__(self,
                 core_env: Union[CoreEnv, dict],
                 action_conversion: CollectionOfConfigType,
                 observation_conversion: CollectionOfConfigType):
        super().__init__(
            core_env=Registry.build_obj(DummyCoreEnvironment, core_env),
            action_conversion_dict=self.action_conversion_registry.arg_to_collection(action_conversion),
            observation_conversion_dict=self.observation_conversion_registry.arg_to_collection(observation_conversion)
        )
