"""An environment interface for space definitions."""
from abc import abstractmethod, ABC
from typing import Union, Dict

import gym


class StructuredEnvSpacesMixin(ABC):
    """This interface complements the StructuredEnv by action and observation spaces.

    StructuredEnv defines the logic and is usually implemented in the core env. In order to make it a complete,
    trainable env, the space definitions from this class are needed.
    """

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Dict:
        """The currently active gym action space.
        """

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        """The currently active gym observation space.
        """

    @property
    @abstractmethod
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym action spaces, with policy IDs as keys.
        """

    @property
    @abstractmethod
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym observation spaces, with policy IDs as keys.
        """
