"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from __future__ import annotations

from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper
from maze.test.core.wrappers.dummy_wrappers.dummy_wrappers import DummyWrapper


class DummyWrapperA(DummyWrapper):
    """
    Dummy wrapper.
    """

    def __init__(self, env: MazeEnv | Wrapper, arg_a: str):
        """
        Initialize dummy wrapper.
        :param env: The inner env.
        :param arg_a: Arbitrary argument.
        """

        super().__init__(env)
        self.arg_a = arg_a

    def do_stuff(self) -> str:
        return self.arg_a
