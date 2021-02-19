"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from typing import Union

import gym

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class DummyWrapperB(Wrapper[Union[BaseEnv, Wrapper]]):
    def __init__(self, env: Union[gym.Env, Wrapper]):
        super().__init__(env)

    def do_stuff(self) -> str:
        return "d"
