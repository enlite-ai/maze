"""Tests for loading config from experiment output directory"""
import os

import pytest
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def train_cartpole():
    """Run training
    :return: The experiment output directory.
    """
    train_hydra_overrides = {
        "algorithm": "ppo",
        "algorithm.n_epochs": "2",
        "algorithm.rollout_evaluator.n_episodes": "0",
        "env": "gym_env",
        "env.name": "CartPole-v1"}

    run_maze_job(train_hydra_overrides, config_module="maze.conf", config_name="conf_train")

    return os.getcwd()


@pytest.fixture(scope="session")
def experiment_out_dir():
    """Train CartPole once per test session and provide the output directory."""
    return train_cartpole()


@pytest.mark.parametrize("use_input_dir_env", [True, False])
@pytest.mark.parametrize("use_input_dir_wrappers", [True, False])
def test_train_and_rollout(experiment_out_dir, use_input_dir_env, use_input_dir_wrappers):
    """Test loading config from the experiment output directory in the rollout run"""
    rollout_hydra_overrides = {
        "runner": "sequential",
        "runner.n_episodes": "2",
        "policy": "torch_policy",
        "+use_input_dir_config.use_input_dir_env": use_input_dir_env,
        "+use_input_dir_config.use_input_dir_wrappers": use_input_dir_wrappers,
        "input_dir": experiment_out_dir,
    }

    run_maze_job(
        rollout_hydra_overrides,
        config_module="maze.conf",
        config_name="conf_rollout"
    )
