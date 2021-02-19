"""Interface specifying the conversion of abstract environment state to the gym-compatible observation."""
from abc import ABC, abstractmethod
from typing import Dict

from gym import spaces
import numpy as np

from maze.core.env.maze_state import MazeStateType

ObservationType = Dict[str, np.ndarray]


class ObservationConversionInterface(ABC):
    """Interface specifying the conversion of abstract environment state to the gym-compatible observation.
    """

    @abstractmethod
    def maze_to_space(self, maze_state: MazeStateType) -> ObservationType:
        """Converts core environment state to a machine readable agent observation.
        """

    def space_to_maze(self, observation: ObservationType) -> MazeStateType:
        """Converts agent observation to core environment state.
        (This is most like not possible for most observation observation_conversion)
        """

    def space(self) -> spaces.Dict:
        """Returns respective Gym observation space.
        """
