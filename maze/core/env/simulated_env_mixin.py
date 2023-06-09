"""Environment interface for simulated environments (used e.g. by Monte Carlo Tree Search)."""
from abc import ABC, abstractmethod
from typing import Any

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

    def serialize_state(self) -> Any:
        """Serialize the current env state and return an object that can be used to deserialize the env again.
        NOTE: The method is optional.
        """
        raise NotImplementedError

    def deserialize_state(self, serialized_state: Any) -> None:
        """Deserialize the current env from the given env state."""
        raise NotImplementedError

