""" Contains observation normalization tests. """
import copy
import os

import numpy as np
import pytest
import maze.test.core.wrappers.observation_normalization as test_observation_normalization_module

from maze.core.agent.random_policy import RandomPolicy
from maze.core.env.structured_env import ActorID
from maze.core.log_stats.log_stats import register_log_stats_writer, increment_log_step
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.log_stats.log_stats_writer_tensorboard import LogStatsWriterTensorboard
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_normalization.normalization_strategies.base import \
    ObservationNormalizationStrategy
from maze.core.wrappers.observation_normalization.observation_normalization_utils import obtain_normalization_statistics
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.test.shared_test_utils.config_testing_utils import load_env_config


def random_env_steps(env: ObservationNormalizationWrapper, steps: int) -> np.ndarray:
    """Randomly interact with environment"""
    observations = []
    obs = env.reset()
    observations.append(obs["observation"])
    for _ in range(steps):
        action = env.sampling_policy.compute_action(obs, maze_state=None, env=env, actor_id=ActorID(0, 0), deterministic=False)
        obs, rew, done, info = env.step(action)
        observations.append(obs["observation"])
        if done:
            obs = env.reset()
            observations.append(obs["observation"])
    return np.vstack(observations)


def run_observation_normalization_pipeline(normalization_config) -> ObservationNormalizationWrapper:
    """ observation normalization test """

    # wrap env with observation normalization
    env = GymMazeEnv("CartPole-v0")
    env = ObservationNormalizationWrapper(env,
                                          default_strategy=normalization_config["default_strategy"],
                                          default_strategy_config=normalization_config["default_strategy_config"],
                                          default_statistics=normalization_config["default_statistics"],
                                          statistics_dump=normalization_config["statistics_dump"],
                                          exclude=normalization_config["exclude"],
                                          sampling_policy=RandomPolicy(env.action_spaces_dict),
                                          manual_config=normalization_config["manual_config"])

    # estimate normalization statistics
    statistics = obtain_normalization_statistics(env, n_samples=1000)

    # check statistics
    for sub_step_key in env.observation_spaces_dict:
        for obs_key in env.observation_spaces_dict[sub_step_key].spaces:
            assert obs_key in statistics
            for stats_key in statistics[obs_key]:
                stats = statistics[obs_key][stats_key]
                assert isinstance(stats, np.ndarray)

    # test normalization
    random_env_steps(env, steps=100)

    return env


def test_observation_normalization_manual_stats():
    """ observation normalization test """

    # init environment
    env = GymMazeEnv("CartPole-v0")

    # manual normalization configs
    normalization_config_1 = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": None},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "sampling_policy": RandomPolicy(env.action_spaces_dict),
        "exclude": None,
        "manual_config": {
            "observation": {
                "strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
                "strategy_config": {"clip_range": (0, 1), "axis": 0},
                "statistics": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}
            }
        }
    }

    normalization_config_2 = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (0, 1), "axis": 0},
        "default_statistics": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
        "statistics_dump": "statistics.pkl",
        "sampling_policy": RandomPolicy(env.action_spaces_dict),
        "exclude": None,
        "manual_config": None,
    }

    def test_config(normalization_config):
        # init environment
        env = GymMazeEnv("CartPole-v0")

        # wrap env with observation normalization
        env = ObservationNormalizationWrapper(env,
                                              default_strategy=normalization_config["default_strategy"],
                                              default_strategy_config=normalization_config["default_strategy_config"],
                                              default_statistics=normalization_config["default_statistics"],
                                              statistics_dump=normalization_config["statistics_dump"],
                                              sampling_policy=normalization_config['sampling_policy'],
                                              exclude=normalization_config["exclude"],
                                              manual_config=normalization_config["manual_config"])

        # check if action space clipping was applied
        assert np.alltrue(env.observation_space["observation"].high <= 1.0)
        assert np.alltrue(env.observation_space["observation"].low >= 0.0)

        # check if stats have been set properly
        statistics = env.get_statistics()
        assert np.all(statistics["observation"]["mean"] == np.zeros(shape=4))
        assert np.all(statistics["observation"]["std"] == np.ones(shape=4))

        # test sampling
        obs = random_env_steps(env, steps=100)
        assert np.min(obs) >= 0 and np.max(obs) <= 1

    test_config(normalization_config_1)
    test_config(normalization_config_2)


def test_observation_normalization_manual_default_stats():
    """ observation normalization test """

    # init environment
    env = GymMazeEnv("CartPole-v0")

    # normalization config
    normalization_config = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (0, 1), "axis": 0},
        "default_statistics": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
        "statistics_dump": "statistics.pkl",
        "sampling_policy": RandomPolicy(env.action_spaces_dict),
        "exclude": None,
        "manual_config": None,
    }

    # wrap env with observation normalization
    env = ObservationNormalizationWrapper(env,
                                          default_strategy=normalization_config["default_strategy"],
                                          default_strategy_config=normalization_config["default_strategy_config"],
                                          default_statistics=normalization_config["default_statistics"],
                                          statistics_dump=normalization_config["statistics_dump"],
                                          sampling_policy=normalization_config['sampling_policy'],
                                          exclude=normalization_config["exclude"],
                                          manual_config=normalization_config["manual_config"])

    # check if action space clipping was applied
    assert np.alltrue(env.observation_space["observation"].high <= 1.0)
    assert np.alltrue(env.observation_space["observation"].low >= 0.0)

    # check if stats have been set properly
    statistics = env.get_statistics()
    assert np.all(statistics["observation"]["mean"] == np.zeros(shape=4))
    assert np.all(statistics["observation"]["std"] == np.ones(shape=4))

    # test sampling
    obs = random_env_steps(env, steps=100)
    assert np.min(obs) >= 0 and np.max(obs) <= 1


def test_observation_normalization_pipeline():
    """ observation normalization test """

    # wrap env with observation normalization
    env = GymMazeEnv("CartPole-v0")
    # normalization config
    normalization_config = {
        "default_strategy": "maze.normalization_strategies.RangeZeroOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": 0},
        "default_statistics": None,
        "sampling_policy": RandomPolicy(env.action_spaces_dict),
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": None
    }
    env = ObservationNormalizationWrapper(env,
                                          default_strategy=normalization_config["default_strategy"],
                                          default_strategy_config=normalization_config["default_strategy_config"],
                                          default_statistics=normalization_config["default_statistics"],
                                          statistics_dump=normalization_config["statistics_dump"],
                                          sampling_policy=normalization_config['sampling_policy'],
                                          exclude=normalization_config["exclude"],
                                          manual_config=normalization_config["manual_config"])

    # check statistics
    statistics = env.get_statistics()
    assert statistics["observation"] is None, statistics

    # check that assertion is thrown
    with pytest.raises(AssertionError):
        random_env_steps(env, steps=1)

    # estimate normalization statistics
    statistics = obtain_normalization_statistics(env, n_samples=1000)

    # check statistics
    for sub_step_key in env.observation_spaces_dict:
        for obs_key in env.observation_spaces_dict[sub_step_key].spaces:
            assert obs_key in statistics
            for stats_key in statistics[obs_key]:
                stats = statistics[obs_key][stats_key]
                assert isinstance(stats, np.ndarray)

    # test normalization
    random_env_steps(env, steps=100)

    # test file dump and loading
    statistics_copy = copy.deepcopy(env.get_statistics())
    assert os.path.exists("statistics.pkl")

    # wrap env with observation normalization
    env = GymMazeEnv("CartPole-v0")
    env = ObservationNormalizationWrapper(env,
                                          default_strategy=normalization_config["default_strategy"],
                                          default_strategy_config=normalization_config["default_strategy_config"],
                                          default_statistics=normalization_config["default_statistics"],
                                          statistics_dump=normalization_config["statistics_dump"],
                                          sampling_policy=normalization_config['sampling_policy'],
                                          exclude=normalization_config["exclude"],
                                          manual_config=normalization_config["manual_config"])

    # check if stats loading worked properly
    statistics = env.get_statistics()
    for _ in env.observation_spaces_dict:
        for obs_key in statistics:
            for stats_key in statistics[obs_key]:
                assert np.all(statistics[obs_key][stats_key] == statistics_copy[obs_key][stats_key])

    # check if stepping works
    random_env_steps(env, steps=100)


def test_observation_normalization_configs():
    """ observation normalization test """

    normalization_config = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": 0},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": None
    }

    run_observation_normalization_pipeline(normalization_config)

    normalization_config = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": 0},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": {
            "observation": {
                "clip_range": (0, 1)
            }
        }
    }

    run_observation_normalization_pipeline(normalization_config)

    normalization_config = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": 0},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": {
            "observation": {
                "strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
                "statistics": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}
            }
        }
    }

    run_observation_normalization_pipeline(normalization_config)


@pytest.mark.parametrize("strategy,axis", [
    (strategy, axis)
    for strategy in [
        "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "maze.normalization_strategies.RangeZeroOneObservationNormalizationStrategy"
    ]
    for axis in [
        0, None
    ]
])
def test_observation_normalization_axis(strategy, axis):
    """ observation normalization test """

    normalization_config = {
        "default_strategy": strategy,
        "default_strategy_config": {"clip_range": (None, None), "axis": axis},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": None
    }

    run_observation_normalization_pipeline(normalization_config)


def test_observation_normalization_init_from_yaml_config():
    """ observation normalization test """

    # load config
    config = load_env_config(test_observation_normalization_module, "dummy_config_file.yml")

    # init environment
    env = GymMazeEnv("CartPole-v0")
    env = ObservationNormalizationWrapper(env, **config["observation_normalization_wrapper"])
    assert isinstance(env, ObservationNormalizationWrapper)

    stats = env.get_statistics()
    assert "stat_1" in stats["observation"] and "stat_2" in stats["observation"]

    norm_strategies = getattr(env, "_normalization_strategies")
    strategy = norm_strategies["observation"]
    assert isinstance(strategy, ObservationNormalizationStrategy)
    assert strategy._clip_min == 0
    assert strategy._clip_max == 1
    assert np.all(strategy._statistics["stat_1"] == np.asarray([0, 0, 0, 0]))
    assert np.all(strategy._statistics["stat_2"] == np.asarray([1, 1, 1, 1]))


def test_observation_statistics_logging():
    """ observation normalization logging test """

    # normalization config
    normalization_config = {
        "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
        "default_strategy_config": {"clip_range": (None, None), "axis": 0},
        "default_statistics": None,
        "statistics_dump": "statistics.pkl",
        "exclude": None,
        "manual_config": {
            "observation": {
                "strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
                "strategy_config": {"clip_range": (0, 1)},
                "statistics": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}
            }
        }
    }
    writer = LogStatsWriterTensorboard(log_dir='test_log', tensorboard_render_figure=True)
    register_log_stats_writer(writer)
    # attach a console writer as well for immediate console feedback
    register_log_stats_writer(LogStatsWriterConsole())

    # init environment
    env = GymMazeEnv("CartPole-v0")

    # wrap env with observation normalization
    env = ObservationNormalizationWrapper(env,
                                          default_strategy=normalization_config["default_strategy"],
                                          default_strategy_config=normalization_config["default_strategy_config"],
                                          default_statistics=normalization_config["default_statistics"],
                                          statistics_dump=normalization_config["statistics_dump"],
                                          sampling_policy=RandomPolicy(env.action_spaces_dict),
                                          exclude=normalization_config["exclude"],
                                          manual_config=normalization_config["manual_config"])

    env = LogStatsWrapper.wrap(env, logging_prefix="train")

    n_episodes = 10
    n_steps_per_episode = 100
    for episode in range(n_episodes):
        _ = env.reset()
        for step in range(n_steps_per_episode):
            # take random action
            action = env.action_space.sample()

            # take step in env and trigger log stats writing
            _, _, done, _ = env.step(action)

            if done:
                break

        increment_log_step()
