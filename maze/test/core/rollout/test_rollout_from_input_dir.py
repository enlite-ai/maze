"""Tests for loading config from experiment output directory"""

from __future__ import annotations

import os

from maze.test.shared_test_utils.run_maze_utils import run_maze_job

import pytest


def train_cartpole():
    """Run training
    :return: The experiment output directory.
    """
    train_hydra_overrides = {
        'algorithm': 'ppo',
        'algorithm.n_epochs': '2',
        'algorithm.rollout_evaluator.n_episodes': '0',
        'env': 'gym_env',
        'env.name': 'CartPole-v1',
        'hydra.run.dir': '.',
    }

    run_maze_job(train_hydra_overrides, config_module='maze.conf', config_name='conf_train')

    return os.getcwd()


@pytest.fixture(scope='session')
def experiment_out_dir(tmp_path_factory):
    """Train CartPole once per test session and provide the output directory."""
    tmp = tmp_path_factory.mktemp('cartpole_train')
    original = os.getcwd()
    os.chdir(tmp)
    try:
        return train_cartpole()
    finally:
        os.chdir(original)  # restore so other session-scoped setup isn't affected


@pytest.mark.parametrize('use_input_dir_env', [True, False])
@pytest.mark.parametrize('use_input_dir_wrappers', [True, False])
@pytest.mark.parametrize('use_input_dir_model', [True, False])
def test_train_and_rollout(experiment_out_dir, use_input_dir_env, use_input_dir_wrappers, use_input_dir_model):
    """Test loading config from the experiment output directory in the rollout run"""
    rollout_hydra_overrides = {
        'runner': 'sequential',
        'runner.n_episodes': '2',
        'policy': 'torch_policy',
        '+use_input_dir_config.use_input_dir_env': use_input_dir_env,
        '+use_input_dir_config.use_input_dir_wrappers': use_input_dir_wrappers,
        '+use_input_dir_config.use_input_dir_model': use_input_dir_model,
        'input_dir': experiment_out_dir,
        'log_base_dir': 'outputs',
    }

    run_maze_job(rollout_hydra_overrides, config_module='maze.conf', config_name='conf_rollout')
