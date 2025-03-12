import os
import time
from typing import Dict, Union, Tuple, Any

import gymnasium as gym
import numpy as np
import pytest

from maze.core.agent.dummy_cartpole_policy import DummyCartPolePolicy
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.env_profiling_events import EnvProfilingEvents
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_writer_logger import LogStatsWriterLogger
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv, GymCoreEnv, GymActionConversion, \
    GymObservationConversion
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


class CustomGymCoreEnv(GymCoreEnv):

    def step(self, maze_action: MazeActionType) -> Tuple[MazeStateType, Union[float, np.ndarray, Any], bool, Dict[Any, Any]]:
        """Intercept ``CoreEnv.step``"""
        self._investigate_step_function_parts = {'main_part': 0, 'other_part': 0}

        cur_time = time.time()
        maze_state, rew, done, info = super().step(maze_action)
        self._investigate_step_function_parts['main_part'] += time.time() - cur_time
        cur_time = time.time()

        # Simulate doing something else.
        time.sleep(0.1)
        self._investigate_step_function_parts['other_part'] += time.time() - cur_time

        return maze_state, rew, done, info


class CustomGymMazeEnv(MazeEnv):
    """Wraps a Gym env into a Maze environment.

    **Example**: *env = GymMazeEnv(env="CartPole-v1")*

    :param env: The gym environment to wrap or the environment id.
    """

    def __init__(self, env: Union[str, gym.Env]):
        if not isinstance(env, gym.Env):
            env = gym.make(env)

        super().__init__(
            core_env=CustomGymCoreEnv(env),
            action_conversion_dict={0: GymActionConversion(env=env)},
            observation_conversion_dict={0: GymObservationConversion(env=env)})


def test_profiling_events_recorded():
    env, agent = GymMazeEnv("CartPole-v1", render_mode=None), DummyCartPolePolicy()

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


def test_profiling_events_recorded_core_env():
    env, agent = CustomGymMazeEnv("CartPole-v1"), DummyCartPolePolicy()
    register_log_stats_writer(LogStatsWriterLogger())
    env = LogStatsWrapper.wrap(env)

    env.seed(1234)
    agent.seed(1235)
    obs = env.reset()
    act = agent.compute_action(obs)

    start_time = time.time()
    env.step(act)
    run_time = time.time() - start_time

    assert env._full_wrapper_stack == ['LogStatsWrapper']
    assert 'action_conversion' in env.profiling_times
    assert 'core_env' in env.profiling_times
    assert 'observation_conversion' in env.profiling_times
    assert hasattr(env, '_investigate_step_function_parts')
    assert 'main_part' in env._investigate_step_function_parts
    assert 'other_part' in env._investigate_step_function_parts
    assert np.isclose(env._investigate_step_function_parts['other_part'], 0.1, atol=0.02)

    assert env._last_profiling_time > sum(env.profiling_times.values())
    assert run_time > env._last_profiling_time

    assert env.step_stats.last_stats[(EnvProfilingEvents.full_env_step_time, 'len', None)] == 1
    assert env.step_stats.last_stats[(EnvProfilingEvents.wrapper_step_time, 'st_mean', ('LogStatsWrapper',))] > 0

    assert env.step_stats.last_stats[(EnvProfilingEvents.investigate_time, 'step_per', ('other_part',))] > 0.99


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
