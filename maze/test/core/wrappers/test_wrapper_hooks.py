from typing import Any
from unittest.mock import patch

from maze.core.env.base_env import BaseEnv
from maze.core.wrappers.wrapper import Wrapper


class _InnerEnv(BaseEnv):
    def seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        pass

    def reset(self) -> Any:
        pass

    def step(self, action):
        pass

    def before_step(self, action):
        pass


class _WrapperWithBeforeStep(Wrapper):
    def before_step(self, action):
        pass

    def step(self, action):
        return action + 1


def test_before_step_1():
    #
    # == test if the nested environment receives the "before_step" hook ==
    #
    with patch.object(_InnerEnv, 'before_step') as before_step_mock:
        env = _InnerEnv()
        env = Wrapper(env)
        step_return = env.step(41)

    assert step_return == 42
    before_step_mock.assert_called_once_with(41)


def test_before_step_2():
    #
    # == test if the Wrapper itself receives the "before_step" hook ==
    #
    with patch.object(_WrapperWithBeforeStep, 'before_step') as before_step_mock:
        env = _InnerEnv()
        env = _WrapperWithBeforeStep(env)
        step_return = env.step(41)

    assert step_return == 42
    before_step_mock.assert_called_once_with(41)


def test_before_step_3():
    #
    # == test if the hook unregister works ==
    #
    with patch.object(_WrapperWithBeforeStep, 'before_step') as before_step_mock:
        env = _InnerEnv()
        env = _WrapperWithBeforeStep(env)
        # hooks for _WrapperWithBeforeStep are active, see if these are unregistered correctly
        env = Wrapper(env)
        step_return = env.step(41)

    assert step_return == 42
    before_step_mock.assert_called_once_with(41)
