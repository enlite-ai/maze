"""Interface common to all environments."""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict


class BaseEnv(ABC):
    """Interface definition for reinforcement learning environments
    defining the minimum required functionality for being considered an environment.
    """

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Environment step function.

        :param action: the selected action to take.
        :return: state, reward, done, info
        """

    @abstractmethod
    def reset(self) -> Any:
        """Resets the environment and returns the initial state.

        :return: the initial state after resetting.
        """

    @abstractmethod
    def seed(self, seed: Any) -> None:
        """Sets the seed for this environment.

        Commonly an integer is sufficient to seed the random number generator(s), but more expressive env-specific
        seed structured are also supported.

        :param: seed: the seed integer initializing the random number generator or an env-specific seed structure.
        """

    @abstractmethod
    def close(self) -> None:
        """Performs any necessary cleanup.
        """
