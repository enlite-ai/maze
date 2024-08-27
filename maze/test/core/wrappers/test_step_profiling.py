import os
import time
from typing import Dict

import pytest

from maze.core.agent.dummy_cartpole_policy import DummyCartPolePolicy
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def test_profiling_events_recorded():
    env, agent = GymMazeEnv("CartPole-v0"), DummyCartPolePolicy()

    env.seed(1234)
    agent.seed(1235)
    obs = env.reset()
    act = agent.compute_action(obs)

    start_time = time.time()
    env.step(act)
    run_time = time.time() - start_time

    assert env._full_wrapper_stack == []
    assert 'action_conversion' in env.profiling_times
    assert 'core_env' in env.profiling_times
    assert 'observation_conversion' in env.profiling_times

    assert env._last_profiling_time > sum(env.profiling_times.values())
    assert run_time > env._last_profiling_time


heuristic_rollouts = [
    {"runner": "sequential"},
    {"runner": "parallel"}
]

# Ensure we are running test configuration and no wrappers (as we do not have the stats
# to initialize observation normalization with)
heuristic_rollouts_defaults = dict(env="gym_env", configuration="test")
heuristic_rollouts = [pytest.param({**heuristic_rollouts_defaults, **r}, id=r['runner']) for r in heuristic_rollouts]


@pytest.mark.parametrize("hydra_overrides", heuristic_rollouts)
def test_heuristic_rollouts(hydra_overrides: Dict):
    """Runs rollout of a dummy policy on cartpole using the sequential and parallel runners."""
    run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_rollout")
    assert os.path.exists('env_profiling.png')
