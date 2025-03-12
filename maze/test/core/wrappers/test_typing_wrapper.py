from abc import ABC
from typing import Any

import gymnasium as gym
import numpy as np

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class _EnvInterfaceInner(ABC):
    def method_inner(self):
        raise NotImplementedError


class _EnvInterfaceWrapper(ABC):
    def method_wrapper(self):
        raise NotImplementedError


class _InnerEnv(gym.Env, _EnvInterfaceInner):
    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def method_inner(self) -> int:
        return 41


class _WrapperWithInterface(gym.Wrapper, _EnvInterfaceWrapper):
    def method_wrapper(self):
        # forward the call to the inner method
        value = self.env.method_inner()

        # just do something
        return value + 1


class _MazeInnerEnv(BaseEnv, _EnvInterfaceInner):
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


class _MazeWrapperWithInterface(Wrapper, _EnvInterfaceWrapper):
    def method_wrapper(self):
        # forward the call to the inner method
        value = self.env.method_inner()

        # just do something
        return value + 1


def test_gym_typing_wrapper():
    #
    # == test if the wrapper correctly mimics isinstance() behaviour ==
    # Note: gym.Wrapper does not allow to pass methods in between them (new with gymnasium)
    #
    env = _InnerEnv()
    assert env.method_inner() == 41

    env = gym.Wrapper(env)

    with np.testing.assert_raises(AttributeError):
        # check if we can still call the inner method
        env.method_inner()

    env = _WrapperWithInterface(env)

    with np.testing.assert_raises(AttributeError):
        # check if we can still call the inner method
        env.method_inner()

    with np.testing.assert_raises(AttributeError):
        # check if we can still call the inner method
        env.method_wrapper()

    env = Wrapper(env)

    with np.testing.assert_raises(AttributeError):
        # check if we can still call the inner method
        env.method_inner()

    with np.testing.assert_raises(AttributeError):
        # check if the wrapper works correctly
        env.method_wrapper()

    assert (
        isinstance(env, _EnvInterfaceWrapper) and
        isinstance(env, _EnvInterfaceInner) and
        isinstance(env, _WrapperWithInterface)
    )


def test_maze_typing_wrapper_is_instance():
    #
    # == test if the wrapper correctly mimics isinstance() behaviour ==
    #
    env = _MazeInnerEnv()
    assert env.method_inner() == 41

    env = _MazeWrapperWithInterface(env)
    assert env.method_inner() == 41
    assert env.method_wrapper() == 42

    env = Wrapper(env)

    assert (
        isinstance(env, _EnvInterfaceWrapper) and
        isinstance(env, _EnvInterfaceInner) and
        isinstance(env, _MazeWrapperWithInterface)
    )

    # check if we can still call the inner method
    assert env.method_inner() == 41

    # check if the wrapper works correctly
    assert env.method_wrapper() == 42

def test_maze_typing_wrapper_idempotency():
    #
    # == test wrapper idempotency ==
    #
    env = _MazeInnerEnv()
    assert env.method_inner() == 41

    env = _MazeWrapperWithInterface(env)
    assert env.method_inner() == 41
    assert env.method_wrapper() == 42

    env = _MazeWrapperWithInterface(env)
    env = _MazeWrapperWithInterface(env)
    env = _MazeWrapperWithInterface(env)

    env = Wrapper(env)

    assert (
        isinstance(env, _EnvInterfaceWrapper) and
        isinstance(env, _EnvInterfaceInner) and
        isinstance(env, _MazeWrapperWithInterface)
    )

    # check if we can still call the inner method
    assert env.method_inner() == 41

    # check if the wrapper works correctly
    assert env.method_wrapper() == 42
