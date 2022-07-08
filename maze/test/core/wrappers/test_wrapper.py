"""Test for general wrapper functionality."""

from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.wrapper import Wrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


class _NestedWrapper(Wrapper[MazeEnv]):
    """Mock wrapper with attributes, for testing attributes assignment in nested scenarios."""

    def __init__(self, env: MazeEnv):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        # BaseEnv is a subset of gym.Env
        super().__init__(env)

        # Attribute present only in this wrapper
        self.custom_attribute = 0

        # Attribute name present in LogStatsWrapper as well
        self.last_env_time = -1


def test_assigning_attributes_across_wrapper_stack():
    """Attributes should be set on the correct wrappers."""

    env = build_dummy_maze_env()
    env = _NestedWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    # -- Custom attribute (present only in the nested wrapper) --

    assert env.custom_attribute == 0
    assert env.env.custom_attribute == 0
    assert not hasattr(env.env.env, "custom_attribute")

    env.custom_attribute = 1

    # The assignment should happen in the nested wrapper
    assert env.custom_attribute == 1
    assert env.env.custom_attribute == 1
    assert not hasattr(env.env.env, "custom_attribute")

    # -- General attribute (present already in the top-level wrapper) --

    assert env.last_env_time is None
    assert env.env.last_env_time == -1
    assert not hasattr(env.env.env, "custom_attribute")

    env.last_env_time = 2

    # The assignment should happen in the top-level wrapper and not bubble down anymore
    assert env.last_env_time == 2  # Only this should get set, as the attribute is present also in log_stats_wrapper
    assert env.env.last_env_time == -1  # This should stay the same as it was before
    assert not hasattr(env.env.env, "custom_attribute")
