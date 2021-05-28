"""Contains cli unit tests."""
import subprocess


def test_cli():
    """Simple blackbox test, run default training for 2 epochs"""
    result = subprocess.run(["maze-run", "-cn", "conf_train", "algorithm.n_epochs=2"])
    assert result.returncode == 0


def test_grid_search():
    """Simple test for the multirun flag used e.g. to run a grid search."""
    result = subprocess.run(["maze-run", "-cn", "conf_train", "configuration=test", "algorithm=ppo",
                             "algorithm.lr=0.0001,0.0005", "+experiment=grid_search", "--multirun"])
    assert result.returncode == 0


def test_nevergrad():
    """Simple test for the nevergrad hyper parameter optimizer."""
    result = subprocess.run(["maze-run", "-cn", "conf_train",
                             "configuration=test", "+experiment=nevergrad",
                             "hydra.sweeper.optim.budget=2", "--multirun"])
    assert result.returncode == 0
