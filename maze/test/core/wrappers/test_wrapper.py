from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.wrapper import Wrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


class _NestedWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that fires test events during reset."""

    def __init__(self, env: MazeEnv):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        # BaseEnv is a subset of gym.Env
        super().__init__(env)

        # Attribute present only in this wrapper
        self.custom_attribute = 0

        # Attribute name present in LogStatsWrapper as well
        self.last_env_time = -1


def test_how_it_currently_works():
    """Note: This test now fails with the "fixed" version"""
    env = build_dummy_maze_env()
    env = _NestedWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    # -- Custom attribute --

    assert env.custom_attribute == 0
    assert env.env.custom_attribute == 0
    assert not hasattr(env.env.env, "custom_attribute")

    env.custom_attribute = 1

    assert env.custom_attribute == 1
    assert env.env.custom_attribute == 0  # This does not get set --> this is the problem
    assert not hasattr(env.env.env, "custom_attribute")

    # -- Attribute present in multiple wrappers --

    assert env.last_env_time is None
    assert env.env.last_env_time == -1
    assert not hasattr(env.env.env, "custom_attribute")

    env.last_env_time = 2

    assert env.last_env_time == 2  # Only this should get set
    assert env.env.last_env_time == -1  # This stays the same as we want it
    assert not hasattr(env.env.env, "custom_attribute")


def test_how_it_should_work():
    env = build_dummy_maze_env()
    env = _NestedWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    # -- Custom attribute --

    assert env.custom_attribute == 0
    assert env.env.custom_attribute == 0
    assert not hasattr(env.env.env, "custom_attribute")

    env.custom_attribute = 1

    assert env.custom_attribute == 1
    assert env.env.custom_attribute == 1  # This now gets set as well
    assert not hasattr(env.env.env, "custom_attribute")

    # -- Attribute present in multiple wrappers --

    assert env.last_env_time is None
    assert env.env.last_env_time == -1
    assert not hasattr(env.env.env, "custom_attribute")

    env.last_env_time = 2

    assert env.last_env_time == 2  # Only this should get set, as the attribute is present in log_stats_wrapper
    assert env.env.last_env_time == -1  # This stays the same as we want it
    assert not hasattr(env.env.env, "custom_attribute")
