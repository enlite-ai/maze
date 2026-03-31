"""Interface specifying the conversion of abstract environment state to the gym-compatible observation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from maze.core.env.maze_state import MazeStateType

import numpy as np
import torch
from gymnasium import spaces

ObservationType = dict[str, np.ndarray]
TorchObservationType = dict[str, torch.Tensor]


class ObservationConversionInterface(ABC):
    """Interface specifying the conversion of abstract environment state to the gym-compatible observation."""

    @abstractmethod
    def maze_to_space(self, maze_state: MazeStateType) -> ObservationType:
        """Converts core environment state to a machine readable agent observation."""

    @abstractmethod
    def space_to_maze(self, observation: ObservationType) -> MazeStateType:
        """Converts agent observation to core environment state.
        (This is most like not possible for most observation observation_conversion)
        """

    @abstractmethod
    def space(self) -> spaces.Dict:
        """Returns respective Gym observation space."""
