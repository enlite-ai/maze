"""Tests related specifically to log_stats_wrapper mechanics (stats and event logging itself is tested separately)"""

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.log_events.monitoring_events import RewardEvents
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.wrapper import Wrapper
from maze.test.shared_test_utils.dummy_wrappers.step_skip_in_reset_wrapper import StepSkipInResetWrapper
from maze.test.shared_test_utils.dummy_wrappers.step_skip_in_step_wrapper import StepSkipInStepWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, build_dummy_structured_env


class _EventsInResetWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that fires test events during reset."""

    def reset(self):
        """Reset the env, then fire the test event (the ordering matters)"""
        obs = self.env.reset()
        base_events = self.core_env.context.event_service.create_event_topic(BaseEnvEvents)
        base_events.test_event(1)
        return obs


def test_records_stats():
    # the default simple setup: flat, single-step env, no step skipping etc.
    env = build_dummy_maze_env()
    env = LogStatsWrapper.wrap(env)

    env.reset()
    for i in range(5):
        env.step(env.action_space.sample())

    # both step counts seen from outside and seen from core env should correspond to 5

    env.write_epoch_stats()
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 5

    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 5


def test_records_events_in_reset():
    env = build_dummy_maze_env()
    env = _EventsInResetWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    env.reset()
    for i in range(5):
        env.step(env.action_space.sample())

    env.write_epoch_stats()
    assert env.get_stats_value(
        BaseEnvEvents.test_event,
        LogStatsLevel.EPOCH
    ) == 1  # only from the single event fired during env reset


def test_records_policy_events():
    env = build_dummy_maze_env()
    env = LogStatsWrapper.wrap(env)

    base_events = env.core_env.context.event_service.create_event_topic(BaseEnvEvents)
    env.reset()
    for i in range(5):
        base_events.test_event(1)  # Simulate firing event from policy (= outside of env.step)
        env.step(env.action_space.sample())

    env.write_epoch_stats()
    assert env.get_stats_value(
        BaseEnvEvents.test_event,
        LogStatsLevel.EPOCH
    ) == 5  # value of 1 x 5 steps


def test_handles_multi_step_setup():
    env = build_dummy_structured_env()
    env = LogStatsWrapper.wrap(env)

    # Step the env four times (should correspond to two core-env steps)
    env.reset()
    for i in range(4):
        env.step(env.action_space.sample())

    # => events should be collected for 2 steps in total
    assert len(env.episode_event_log.step_event_logs) == 2

    # The same goes for both reward stats from outside and from core-env perspective

    env.write_epoch_stats()
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 2

    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 2


def test_handles_step_skipping_in_reset():
    env = build_dummy_maze_env()
    env = StepSkipInResetWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    env.reset()
    # Step the env once (should be the third step -- first two were done in the reset)
    env.step(env.action_space.sample())

    # Events should be collected for 3 steps in total -- two from the env reset done by the wrapper + one done above
    assert len(env.episode_event_log.step_event_logs) == 3

    # The same goes for "original reward" stats
    env.write_epoch_stats()
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 3

    # The step count from outside is still one (as normal reward events should not be fired for "skipped" steps)
    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 1


def test_handles_step_skipping_in_step():
    env = build_dummy_maze_env()
    env = StepSkipInStepWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    # Step the env twice (should correspond to four core-env steps)
    env.reset()
    for i in range(2):
        env.step(env.action_space.sample())

    # => events should be collected for 4 steps in total
    assert len(env.episode_event_log.step_event_logs) == 4

    # The same goes for "original reward" stats
    env.write_epoch_stats()
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 4

    # The step count from outside is still just two (as normal reward events should not be fired for "skipped" steps)
    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 2
