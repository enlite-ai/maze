"""Test rollouts run using the rollout runners"""

from typing import Dict

import pytest

from maze.core.agent.dummy_cartpole_policy import DummyCartPolePolicy
from maze.core.rollout.parallel_rollout_runner import ParallelRolloutRunner
from maze.core.rollout.sequential_rollout_runner import SequentialRolloutRunner
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.monitoring_wrapper import MazeEnvMonitoringWrapper
from maze.test.shared_test_utils.run_maze_utils import run_maze_job

heuristic_rollouts = [
    {"runner": "sequential"},
    {"runner": "parallel"}
]

# Ensure we are running test configuration and no wrappers (as we do not have the stats
# to initialize observation normalization with)
heuristic_rollouts_defaults = dict(env="gym_env", configuration="test")
heuristic_rollouts = [pytest.param({**heuristic_rollouts_defaults, **r}, id=r['runner']) for r in heuristic_rollouts]


@pytest.mark.parametrize("hydra_overrides", heuristic_rollouts)
def test_heuristic_rollouts(hydra_overrides: Dict[str, str]):
    """Runs rollout of a dummy policy on cartpole using the sequential and parallel runners."""
    run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_rollout")


def test_rollouts_from_python():
    env, agent = GymMazeEnv("CartPole-v0"), DummyCartPolePolicy()

    sequential = SequentialRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        record_trajectory=False,
        record_event_logs=False,
        render=False)
    sequential.maze_seeding = MazeSeeding(1234, 4321, False)
    sequential.run_with(env=env, wrappers={}, agent=agent)

    parallel = ParallelRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        record_trajectory=False,
        record_event_logs=False,
        n_processes=2)
    parallel.maze_seeding = MazeSeeding(1234, 4321, False)
    # Test with a wrapper config as well
    parallel.run_with(
        env=env,
        wrappers={MazeEnvMonitoringWrapper: {"observation_logging": True,
                                             "action_logging": False,
                                             "reward_logging": False}},
        agent=agent)


def test_sequential_rollout_with_rendering():
    env, agent = GymMazeEnv("CartPole-v0"), DummyCartPolePolicy()
    sequential = SequentialRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        record_trajectory=True,
        record_event_logs=False,
        render=True)
    sequential.maze_seeding = MazeSeeding(1234, 4321, False)
    sequential.run_with(env=env, wrappers={}, agent=agent)
