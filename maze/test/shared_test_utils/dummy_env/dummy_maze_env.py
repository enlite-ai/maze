"""
Includes the implementation of the dummy flat environment.
"""

from typing import Union

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.factory import Factory, CollectionOfConfigType


class DummyEnvironment(MazeEnv):
    """
    Environment.
    """

    def __init__(self,
                 core_env: Union[CoreEnv, dict],
                 action_conversion: CollectionOfConfigType,
                 observation_conversion: CollectionOfConfigType):
        super().__init__(
            core_env=Factory(CoreEnv).instantiate(core_env),
            action_conversion_dict=Factory(ActionConversionInterface).instantiate_collection(action_conversion),
            observation_conversion_dict=Factory(ObservationConversionInterface).instantiate_collection(
                observation_conversion)
        )
