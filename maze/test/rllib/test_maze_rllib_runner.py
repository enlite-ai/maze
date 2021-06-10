"""Test methods for the maze-rllib runner"""
import os
import shutil

import pytest
from gym.envs.classic_control import CartPoleEnv
import ray
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.tune.registry import RLLIB_MODEL, RLLIB_ACTION_DIST, _global_registry

from maze.core.utils.factory import Factory
from maze.rllib.maze_rllib_action_distribution import MazeRLlibActionDistribution
from maze.rllib.maze_rllib_models.maze_rllib_policy_model import MazeRLlibPolicyModel
from maze.rllib.maze_rllib_runner import MazeRLlibRunner
from maze.runner import Runner
from maze.test.shared_test_utils.hydra_helper_functions import load_hydra_config, run_maze_from_str

SPACE_CONFIG_DUMP_FILE = 'space_config_dump.pkl'


@pytest.mark.rllib
def test_runner_basic():
    """Test the basic runner """
    runner = MazeRLlibRunner(SPACE_CONFIG_DUMP_FILE, normalization_samples=5,
                             num_workers=1, tune_config={'training_iteration': 1},
                             ray_config={'local_mode': True},
                             state_dict_dump_file='state_dict.pt')
    assert isinstance(runner, Runner)


@pytest.mark.rllib
def test_init_cartpole_rllib_model():
    """test the init methods"""
    hydra_overrides = {'rllib/runner': 'dev', 'model': 'rllib'}

    cfg = load_hydra_config('maze.conf', 'conf_rllib', hydra_overrides)

    runner = Factory(base_type=MazeRLlibRunner).instantiate(cfg.runner)
    runner.setup(cfg)
    ray_config, rllib_config, tune_config = runner.ray_config, runner.rllib_config, runner.tune_config

    assert isinstance(runner.env_factory(), CartPoleEnv)

    assert isinstance(ray_config, dict)
    assert isinstance(rllib_config, dict)
    assert isinstance(tune_config, dict)

    assert rllib_config['env'] == 'maze_env'
    assert rllib_config['framework'] == 'torch'
    assert rllib_config['num_workers'] == 1
    for k, v in rllib_config['model'].items():
        if v == "DEPRECATED_VALUE":
            v = DEPRECATED_VALUE
        assert k in MODEL_DEFAULTS, f'Maze RLlib model parameter \'{k}\' not in RLlib MODEL_DEFAULTS (rllib version: ' \
                                    f'{ray.__version__})'
        assert MODEL_DEFAULTS[k] == v, f'Rllib key:\'{k}\',value:\'{MODEL_DEFAULTS[k]}\' does not match with the ' \
                                       f'maze defined config \'{v}\' with rllib version: {ray.__version__}'

    if 'ObservationNormalizationWrapper' in cfg.wrappers:
        assert os.path.exists(cfg.wrappers.ObservationNormalizationWrapper.statistics_dump)
        os.remove(cfg.wrappers.ObservationNormalizationWrapper.statistics_dump)


@pytest.mark.rllib
def test_init_cartpole_maze_model():
    """test the init methods """
    hydra_overrides = {'rllib/runner': 'dev', 'configuration': 'test',
                       'env': 'gym_env', 'model': 'vector_obs', 'wrappers': 'vector_obs', 'critic': 'template_state'}

    cfg = load_hydra_config('maze.conf', 'conf_rllib', hydra_overrides)

    runner = Factory(base_type=MazeRLlibRunner).instantiate(cfg.runner)
    runner.setup(cfg)
    ray_config, rllib_config, tune_config = runner.ray_config, runner.rllib_config, runner.tune_config

    assert isinstance(runner.env_factory(), CartPoleEnv)

    assert issubclass(_global_registry.get(RLLIB_ACTION_DIST, 'maze_dist'), MazeRLlibActionDistribution)
    assert issubclass(_global_registry.get(RLLIB_MODEL, 'maze_model'), MazeRLlibPolicyModel)

    assert isinstance(ray_config, dict)
    assert isinstance(rllib_config, dict)
    assert isinstance(tune_config, dict)

    assert rllib_config['env'] == 'maze_env'
    assert rllib_config['framework'] == 'torch'
    assert rllib_config['num_workers'] == 1
    model_config = rllib_config['model']

    assert model_config['custom_action_dist'] == 'maze_dist'
    assert model_config['custom_model'] == 'maze_model'
    assert model_config['vf_share_layers'] is False
    assert model_config['custom_model_config']['maze_model_composer_config'] == cfg.model
    assert model_config['custom_model_config']['spaces_config_dump_file'] == cfg.runner.spaces_config_dump_file

    if 'ObservationNormalizationWrapper' in cfg.wrappers:
        assert os.path.exists(cfg.wrappers.ObservationNormalizationWrapper.statistics_dump)
        os.remove(cfg.wrappers.ObservationNormalizationWrapper.statistics_dump)


@pytest.mark.rllib
def test_run_cartpole():
    """Test full run with cartpole"""
    hydra_overrides = {'rllib/runner': 'dev', 'configuration': 'test',
                       'env': 'gym_env', 'model': 'vector_obs', 'wrappers': 'vector_obs', 'critic': 'template_state'}

    cfg = run_maze_from_str('maze.conf', 'conf_rllib', hydra_overrides)

    assert os.path.exists(cfg.algorithm.algorithm)
    shutil.rmtree(cfg.algorithm.algorithm)


@pytest.mark.rllib
def test_run_cartpole_dqn():
    """Test full run with cartpole"""
    hydra_overrides = {'rllib/runner': 'dev', 'configuration': 'test', 'rllib/algorithm': 'dqn',
                       'env': 'gym_env', 'model': 'vector_obs', 'wrappers': 'vector_obs', 'critic': 'template_state'}

    cfg = run_maze_from_str('maze.conf', 'conf_rllib', hydra_overrides)

    assert os.path.exists(cfg.algorithm.algorithm)
    shutil.rmtree(cfg.algorithm.algorithm)
