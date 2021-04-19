"""Utils for running rollouts through rollout runners in tests."""
import subprocess
from typing import Dict

from hydra.core.hydra_config import HydraConfig
from hydra.experimental import initialize_config_module, compose
from omegaconf import open_dict, DictConfig

from maze.maze_cli import maze_run


def run_maze_job(hydra_overrides: Dict[str, str], config_module: str, config_name: str) -> DictConfig:
    """Runs rollout with the given config overrides using maze_run.

    :param hydra_overrides: Config overrides for hydra.
    :param config_module: The config module.
    :param config_name: The name of the default config.
    """
    with initialize_config_module(config_module=config_module):
        # Config is relative to a module
        # For the HydraConfig init below, we need the hydra key there as well (=> return_hydra_config=True)
        cfg = compose(config_name=config_name,
                      overrides=[key + "=" + str(val) for key, val in hydra_overrides.items()],
                      return_hydra_config=True)

        # Init the HydraConfig: This is when Hydra actually creates the output dir and changes into it
        # (otherwise we only have the config object, but not the full run environment)
        HydraConfig.instance().set_config(cfg)

        # For the rollout itself, the Hydra config should not be there anymore
        with open_dict(cfg):
            del cfg["hydra"]

        # Run the rollout
        maze_run(cfg)

    return cfg


def run_maze_job_through_cli(hydra_overrides: Dict[str, str], config_name: str):
    """Runs rollout with the given config overrides using maze_run in a separate process.

    Note that run this way, Hydra will create an output sub-directory.

    :param hydra_overrides: Config overrides for hydra.
    :param config_name: The name of the default config.
    """

    overrides = [key + "=" + str(val) for key, val in hydra_overrides.items()]
    result = subprocess.run(["maze-run", "-cn", config_name] + overrides)
    assert result.returncode == 0
