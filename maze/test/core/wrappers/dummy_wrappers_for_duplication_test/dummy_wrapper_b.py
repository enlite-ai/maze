"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from __future__ import annotations

from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper


class DummyWrapperB(Wrapper[BaseEnv | Wrapper]):
    def __init__(self, env: MazeEnv | Wrapper):
        super().__init__(env)

    def do_stuff(self) -> str:
        return 'd'
