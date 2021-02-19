"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from abc import abstractmethod, ABC
from typing import Union

import gym

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class DummyWrapper(Wrapper[Union[BaseEnv, Wrapper]], ABC):
    def __init__(self, env: Union[gym.Env, Wrapper]):
        super().__init__(env)

    @abstractmethod
    def do_stuff(self) -> str:
        raise NotImplementedError
