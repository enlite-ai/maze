"""
Dummy wrappers for generic wrapper configuration's unit tests.
"""

from typing import Union

import gym

from maze.core.wrappers.wrapper import Wrapper
from maze.test.core.wrappers.dummy_wrappers.dummy_wrappers import DummyWrapper


class DummyWrapperB(DummyWrapper):
    """
    Dummy wrapper.
    """

    def __init__(
            self,
            env: Union[gym.Env, Wrapper],
            arg_b: str,
            arg_c: str
    ):
        """
        Initialize dummy wrapper.
        :param env: The inner env.
        :param arg_b: Arbitrary argument.
        :param arg_c: Arbitrary argument.
        """

        super().__init__(env)
        self.arg_b = arg_b
        self.arg_c = arg_c

    def do_stuff(self) -> str:
        return self.arg_b
