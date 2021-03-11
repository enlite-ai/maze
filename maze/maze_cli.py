"""Implements the Maze command line interface for running rollouts, trainings and else."""
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from maze.core.utils.factory import Factory
from maze.runner import Runner


@hydra.main(config_path="conf", config_name="conf_rollout")
def maze_run(cfg: DictConfig) -> None:
    """
    Run a CLI task based on the provided configuration.

    A runner object is instantiated according to the config (cfg.runner) and it is then handed
    the whole configuration object (cfg). Runners can perform various tasks such as rollouts, trainings etc.

    :param cfg: Hydra configuration for the rollout.
    """

    print(yaml.dump(OmegaConf.to_container(cfg, resolve=True)))
    runner = Factory(base_type=Runner).instantiate(cfg.runner)
    runner.run(cfg)


if __name__ == "__main__":
    maze_run()
