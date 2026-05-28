"""Test observation visualization wrapper"""

from __future__ import annotations

from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_visualization_wrapper import ObservationVisualizationWrapper
from maze.utils.log_stats_utils import SimpleStatsLoggingSetup


def test_observation_monitoring():
    """Observation logging unit test"""
    env = GymMazeEnv(env='CartPole-v1', render_mode=None)

    env = ObservationVisualizationWrapper.wrap(env, plot_function=None)
    env = LogStatsWrapper.wrap(env, logging_prefix='train')

    terminated, truncated = False, False
    with SimpleStatsLoggingSetup(env, log_dir='.'):
        env.reset()
        while not (terminated or truncated):
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())


def test_observation_visualization_wrapper_none_passthrough():
    """None observations must be returned as None without raising."""
    base_env = GymMazeEnv(env='CartPole-v1', render_mode=None)
    env = ObservationVisualizationWrapper.wrap(base_env, plot_function=None)

    inner_env = env.env
    original_step = inner_env.step

    def patched_step(action):
        _, reward, terminated, truncated, info = original_step(action)
        return None, reward, terminated, truncated, info

    env.reset()
    inner_env.step = patched_step
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs is None
