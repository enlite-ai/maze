"""Environment interface for simulated environments (used e.g. by Monte Carlo Tree Search)."""
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict

from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import StructuredEnv


class SimulatedEnvMixin(ABC):
    """Environment interface for simulated environments.

    The main addition to StructuredEnv is the clone method, which resets the simulation to the given env state.
    This interface is used by Monte Carlo Tree Search."""

    @abstractmethod
    def clone_from(self, maze_state: MazeStateType) -> None:
        """Clone an environment by resetting the simulation to its current state."""

    def step_without_observation(self, action: ActionType) -> Tuple[Any, bool, Dict[Any, Any]]:
        """Environment step function that does not return any observation.

        This method can be significantly faster than the full step function in cases with expensive state to
        observation mappings.

        :param action: the selected action to take.
        :return: reward, done, info
        """

        assert isinstance(self, StructuredEnv)
        obs, reward, done, info = self.step(action)
        return reward, done, info
