"""Environment interface for simulated environments (used e.g. by Monte Carlo Tree Search)."""
from abc import ABC, abstractmethod

from maze.core.env.structured_env import StructuredEnv


class SimulatedEnvMixin(ABC):
    """Environment interface for simulated environments.

    The main addition to StructuredEnv is the clone method, which resets the simulation to the given env state.
    This interface is used by Monte Carlo Tree Search."""

    @abstractmethod
    def clone_from(self, env: StructuredEnv) -> None:
        """Clone an environment by resetting the simulation to its current state.

        :param env: The environment to clone.
        """

