from abc import ABC
from typing import Any

import gym

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class _EnvInterfaceInner(ABC):
    def method_inner(self):
        raise NotImplementedError


class _EnvInterfaceWrapper(ABC):
    def method_wrapper(self):
        raise NotImplementedError


class _InnerEnv(BaseEnv, _EnvInterfaceInner):
    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        pass

    def reset(self) -> Any:
        pass

    def step(self, action):
        pass

    def method_inner(self) -> int:
        return 41


class _WrapperWithInterface(gym.Wrapper, _EnvInterfaceWrapper):
    def method_wrapper(self):
        # forward the call to the inner method
        value = self.env.method_inner()

        # just do something
        return value + 1


def test_typing_wrapper():
    #
    # == test if the wrapper correctly mimics isinstance() behaviour ==
    #
    env = _InnerEnv()
    env = gym.Wrapper(env)
    env = _WrapperWithInterface(env)
    env = Wrapper(env)

    assert (isinstance(env, _EnvInterfaceWrapper) and
            isinstance(env, _EnvInterfaceInner) and
            isinstance(env, _WrapperWithInterface))

    # check if we can still call the inner method
    assert env.method_inner() == 41

    # check if the wrapper works correctly
    assert env.method_wrapper() == 42

    #
    # == test idempotency ==
    #
    env = Wrapper(env)

    assert (isinstance(env, _EnvInterfaceWrapper) and
            isinstance(env, _EnvInterfaceInner) and
            isinstance(env, _WrapperWithInterface))

    # check if we can still call the inner method
    assert env.method_inner() == 41

    # check if the wrapper works correctly
    assert env.method_wrapper() == 42
