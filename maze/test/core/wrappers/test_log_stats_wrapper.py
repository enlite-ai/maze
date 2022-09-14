"""Tests related specifically to log_stats_wrapper mechanics (stats and event logging itself is tested separately)"""

import pytest

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
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


class _StepSkippingAndErrorInResetWrapper(Wrapper[MazeEnv]):
    """Performs step skipping and then raises an error in reset function."""

    def reset(self) -> ObservationType:
        """Skip one step, then raise an error."""
        self.env.reset()
        self.env.step(self.env.noop_action())
        raise RuntimeError("Test Error")


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


def test_counts_episodes_that_skip_and_error_in_reset():
    """Test that episodes where step-skipping during reset is performed and then
    an error is raised (still as part of reset) are still counted as valid episodes."""
    env = build_dummy_maze_env()
    env = _StepSkippingAndErrorInResetWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)

    # Reset the env: This should raise an exception
    with pytest.raises(RuntimeError):
        env.reset()

    # Events should be collected for 1 steps in total (the one step skipped in the reset)
    assert len(env.episode_event_log.step_event_logs) == 1

    env.write_epoch_stats()

    # The original_reward stats should be recorded for the one skipped step
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 1

    # Both episode_count and total_episode_count stats for the reward_original event reflect this
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="episode_count") == 1
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_episode_count") == 1

    # No ordinary "reward" event is counted, as no step was done from the outside
    with pytest.raises(KeyError):
        env.get_stats_value(
            BaseEnvEvents.reward,
            LogStatsLevel.EPOCH,
            name="total_step_count")
