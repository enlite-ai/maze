"""Maze environment tests."""

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env_with_structured_core_env, \
    build_dummy_maze_env


def test_step_increment_in_single_step_core_env():
    """In single sub-step envs, events should be cleared out and env time incremented automatically."""
    env = build_dummy_maze_env()
    env = LogStatsWrapper.wrap(env)

    env.reset()
    assert env.get_env_time() == 0

    # 10 steps
    for _ in range(10):
        env.step(env.action_space.sample())

    assert env.get_env_time() == 10
    env.reset()

    increment_log_step()

    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 10


def test_step_increment_in_structured_core_environments():
    """Structured core envs manage the step incrementing themselves and Maze env should not interfere with that."""
    env = build_dummy_maze_env_with_structured_core_env()
    env = LogStatsWrapper.wrap(env)

    env.reset()
    assert env.get_env_time() == 0

    # Do 10 agent steps => 5 structured steps (as we have two agents)
    for _ in range(10):
        env.step(env.action_space.sample())

    assert env.get_env_time() == 5
    env.reset()

    increment_log_step()

    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 5
