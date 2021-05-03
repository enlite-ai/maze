import subprocess


def test_cli():
    """Simple blackbox test, run default training for 2 epochs"""
    result = subprocess.run(["maze-run", "-cn", "conf_train", "algorithm.n_epochs=2"])
    assert result.returncode == 0
