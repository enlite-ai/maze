"""Tests related specifically to log_stats_wrapper mechanics (stats and event logging itself is tested separately)"""

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.log_events.monitoring_events import RewardEvents
from maze.core.log_stats.log_stats import LogStatsLevel, increment_log_step
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.wrapper import Wrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


class _StepInResetWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that steps the env in the reset function"""

    def reset(self):
        """Step the env twice during the reset function"""
        obs = self.env.reset()
        for i in range(2):
            obs, _, _, _ = self.step(self.env.action_space.sample())
        return obs


def test_allows_stepping_in_reset():
    env = build_dummy_maze_env()
    env = _StepInResetWrapper.wrap(env)
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


def test_records_policy_stats():
    env = build_dummy_maze_env()
    env = LogStatsWrapper.wrap(env)

    base_events = env.core_env.context.event_service.create_event_topic(BaseEnvEvents)
    env.reset()
    for i in range(5):
        base_events.test_event(1)  # Simulate firing event from policy (= outside of env.step)
        _ = env.step(env.action_space.sample())

    env.reset()
    increment_log_step()

    assert env.get_stats_value(
        BaseEnvEvents.test_event,
        LogStatsLevel.EPOCH
    ) == 5  # value of 1 x 5 steps
