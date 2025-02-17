"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from typing import Union

from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper


class DummyWrapperB(Wrapper[Union[BaseEnv, Wrapper]]):
    def __init__(self, env: Union[MazeEnv, Wrapper]):
        super().__init__(env)

    def do_stuff(self) -> str:
        return "d"
