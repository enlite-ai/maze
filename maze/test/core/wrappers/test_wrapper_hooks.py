from typing import Any
from unittest.mock import patch

from maze.core.env.environment_context import EnvironmentContext

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class _InnerEnv(BaseEnv):
    def __init__(self):
        self.context = EnvironmentContext()

    def seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        pass

    def reset(self) -> Any:
        pass

    def step(self, action):
        return 123

    def before_step(self, action):
        pass


class _WrapperWithStep(Wrapper):
    def step(self, action):
        return action + 1


def test_pre_step_1():
    #
    # == test if the EnvironmentContext receives the "pre_step" hook ==
    #
    with patch.object(EnvironmentContext, 'pre_step') as pre_step_mock:
        env = _InnerEnv()
        env = Wrapper(env)
        step_return = env.step(41)

    assert step_return == 123
    pre_step_mock.assert_called_once_with()


def test_pre_step_2():
    #
    # == test if the Wrapper itself triggers the "pre_step" hook ==
    #
    with patch.object(EnvironmentContext, 'pre_step') as pre_step_mock:
        env = _InnerEnv()
        env = _WrapperWithStep(env)
        step_return = env.step(41)

    assert step_return == 42
    pre_step_mock.assert_called_once_with()


def test_pre_step_3():
    #
    # == test if the hook unregister works ==
    #
    with patch.object(EnvironmentContext, 'pre_step') as pre_step_mock:
        env = _InnerEnv()
        env = _WrapperWithStep(env)
        # hooks for _WrapperWithBeforeStep are active, see if these are unregistered correctly
        env = Wrapper(env)
        step_return = env.step(41)

    assert step_return == 42
    pre_step_mock.assert_called_once_with()
