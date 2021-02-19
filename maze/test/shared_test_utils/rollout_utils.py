"""Utils for running rollouts through rollout runners in tests."""

from typing import Dict

from hydra.core.hydra_config import HydraConfig
from hydra.experimental import initialize_config_module, compose
from omegaconf import open_dict

from maze.maze_cli import maze_run


def run_rollout(hydra_overrides: Dict[str, str]):
    """Runs rollout with the given config overrides using maze_run.

    :param hydra_overrides: Config overrides for hydra
    """
    with initialize_config_module(config_module="maze.conf"):
        # Config is relative to a module
        # For the HydraConfig init below, we need the hydra key there as well (=> return_hydra_config=True)
        cfg = compose(config_name="conf_rollout",
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
