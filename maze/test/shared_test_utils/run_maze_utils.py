"""Utils for running rollouts through rollout runners in tests."""

from __future__ import annotations

import os
import subprocess

from maze.core.utils.config_utils import get_hydra_version_base
from maze.maze_cli import maze_run

from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


def run_maze_job(hydra_overrides: dict[str, str], config_module: str, config_name: str) -> DictConfig:
    """Run a maze job with the given Hydra config overrides outside of the Hydra launcher using maze_run.

    This method replicates the behavior of launching a maze job via the CLI (e.g. `maze-run`), but allows programmatic
    invocation from Python code such as tests. It manually handles the steps that the Hydra launcher would normally
    perform:
      1. Composing the config from the given module and overrides.
      2. Initializing the HydraConfig with the composed config.
      3. Creating the output directory and changing into it (replicating hydra.job.chdir: true).
      4. Restoring the original working directory after the job completes, even if it raises.

    Note: Must be called from within a 'if __name__ == '__main__':' guard when using multiprocessing, as the forkserver
    will re-execute the main module when spawning worker processes, causing run_maze_job to be called again for each
    worker otherwise.

    :param hydra_overrides: Dict of Hydra config overrides, e.g. {'env': 'cartpole', 'runner.concurrency': '2'}.
    :param config_module: The Python module containing the config files, e.g. 'maze.conf'.
    :param config_name: The name of the default config file (without .yaml extension), e.g. 'conf_train'.
    :return: The composed DictConfig used for the job (without the hydra key).
    """
    kwargs: dict[str, str | None] = get_hydra_version_base()
    with initialize_config_module(config_module=config_module, **kwargs):
        # Compose the config from the given module and overrides
        # return_hydra_config=True is required for HydraConfig.instance().set_config() below
        cfg = compose(
            config_name=config_name,
            overrides=[key + '=' + str(val) for key, val in hydra_overrides.items()],
            return_hydra_config=True,
        )

        # Init the HydraConfig: This is when Hydra actually creates the output dir and changes into it
        # (otherwise we only have the config object, but not the full run environment)
        HydraConfig.instance().set_config(cfg)

        # Resolve the output directory from the hydra run dir config and change into it, replicating the behavior of
        # 'hydra.job.chdir: true' which only applies when launching via the Hydra CLI launcher
        original_cwd = os.getcwd()

        output_dir = os.path.abspath(OmegaConf.select(cfg, 'hydra.run.dir', default='.'))
        os.makedirs(output_dir, exist_ok=True)

        os.chdir(output_dir)

        # Remove the hydra key from the config before passing it to maze_run, since maze_run expects a plain config
        with open_dict(cfg):
            del cfg['hydra']

        try:
            maze_run(cfg)
        finally:
            # Always restore the original working directory, even if maze_run raises an exception
            os.chdir(original_cwd)

    return cfg


def run_maze_job_through_cli(hydra_overrides: dict[str, str], config_name: str):
    """Runs rollout with the given config overrides using maze_run in a separate process.

    Note that run this way, Hydra will create an output sub-directory.

    :param hydra_overrides: Config overrides for hydra.
    :param config_name: The name of the default config.
    """

    overrides = [key + '=' + str(val) for key, val in hydra_overrides.items()]
    result = subprocess.run(['maze-run', '-cn', config_name] + overrides)
    assert result.returncode == 0
