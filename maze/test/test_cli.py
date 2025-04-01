"""Contains cli unit tests."""
import subprocess

import pytest
from hydra import initialize_config_module, compose

from maze.maze_cli import maze_run


def test_cli():
    """Simple blackbox test, run default training for 2 epochs"""
    result = subprocess.run(["maze-run", "-cn", "conf_train", "algorithm.n_epochs=2"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_grid_search():
    """Simple test for the multirun flag used e.g. to run a grid search."""
    result = subprocess.run(
        ["maze-run", "-cn", "conf_train", "configuration=test", "algorithm=ppo",
         "algorithm.lr=0.0001,0.0005", "+experiment=grid_search", "--multirun"],
        capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8")


pytest.importorskip("torch_scatter", reason="No module named 'torch_scatter'")
def test_nevergrad():
    """Simple test for the nevergrad hyper parameter optimizer."""
    result = subprocess.run(["maze-run", "-cn", "conf_train",
                             "configuration=test", "+experiment=nevergrad",
                             "hydra.sweeper.optim.budget=2", "--multirun",
                             "hydra.job.chdir=True"],
                            capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8")

    # def read_hydra_config_with_overrides(config_module: str, config_name: str, overrides: list[str]):
    #     """Read and assemble a hydra config, given the config module, name, and overrides.
    #
    #     :param config_module: Python module path of the hydra configuration package
    #     :param config_name: Name of the defaults configuration yaml file within `config_module`
    #     :param overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    #     :return: Hydra DictConfig instance, assembled according to the given module, name, and overrides.
    #     """
    #     with initialize_config_module(config_module):
    #         cfg = compose(config_name, overrides=overrides)
    #
    #     return cfg
    #
    # cfg = read_hydra_config_with_overrides(
    #     config_module='maze.conf',
    #     config_name='conf_train',
    #     overrides=[
    #         'configuration=test',
    #         '+experiment=nevergrad',
    #         'hydra.sweeper.optim.budget=2',
    #         'hydra.job.chdir=True',
    #     ],
    # )
    #
    # maze_run(cfg)
    #
    # print('')
