"""Contains cli unit tests."""

from __future__ import annotations

import subprocess


def test_cli():
    """Simple blackbox test, run default training for 2 epochs"""
    result = subprocess.run(
        [
            'maze-run',
            '-cn',
            'conf_train',
            'algorithm.n_epochs=2',
            'seeding.env_base_seed=1234',
            'seeding.agent_base_seed=1234',
            'log_base_dir=output',
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode('utf-8')


def test_grid_search():
    """Simple test for the multirun flag used e.g. to run a grid search."""
    result = subprocess.run(
        [
            'maze-run',
            '-cn',
            'conf_train',
            'configuration=test',
            'algorithm=ppo',
            'seeding.env_base_seed=1234',
            'seeding.agent_base_seed=1234',
            'algorithm.lr=0.0001,0.0005',
            '+experiment=grid_search',
            'log_base_dir=output',
            '--multirun',
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode('utf-8')


def test_optuna():
    """Simple test for the optuna hyper parameter optimizer."""
    result = subprocess.run(
        [
            'maze-run',
            '-cn',
            'conf_train',
            'algorithm.n_epochs=2',
            'configuration=test',
            '+experiment=optuna',
            'hydra.sweeper.sampler.n_startup_trials=0',
            'hydra.sweeper.n_trials=2',
            'hydra.sweeper.sampler.seed=1234',
            'log_base_dir=output',
            '--multirun',
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode('utf-8')
