"""Implements the Maze command line interface for running rollouts, trainings and else."""
import glob
import os
from typing import Optional

import hydra
import yaml
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from maze.core.utils.factory import Factory
from maze.runner import Runner
from maze.utils.log_stats_utils import clear_global_state
from maze.utils.tensorboard_reader import tensorboard_to_pandas


@hydra.main(config_path="conf", config_name="conf_rollout")
def maze_run(cfg: DictConfig) -> Optional[float]:
    """
    Run a CLI task based on the provided configuration.

    A runner object is instantiated according to the config (cfg.runner) and it is then handed
    the whole configuration object (cfg). Runners can perform various tasks such as rollouts, trainings etc.

    :param cfg: Hydra configuration for the rollout.
    """

    # required for Hydra --multirun (e.g., sweeper)
    clear_global_state()

    def run_job() -> None:
        print(yaml.dump(OmegaConf.to_container(cfg, resolve=True)))
        runner = Factory(base_type=Runner).instantiate(cfg.runner)
        runner.run(cfg)

    f = open('maze_cli.log', 'a+')
    f.write(f'Entering maze_run in directory {os.getcwd()}\n')
    f.close()

    # check if we are currently in a --multirun
    instance = HydraConfig.instance()
    is_multi_run = False if instance.cfg.hydra.job.get("num") is None else True

    # regular single runs
    if not is_multi_run:
        run_job()
    # multi-runs (e.g., gird search, nevergrad, ...)
    else:

        try:
            # run job
            print(yaml.dump(OmegaConf.to_container(cfg, resolve=True)))
            runner = Factory(base_type=Runner).instantiate(cfg.runner)
            runner.run(cfg)

        # when optimizing hyper parameters a single exception
        # in one job should not break the entire experiment
        except:
            return np.finfo(np.float32).min

        # load tensorboard log and return maximum mean reward
        # load tensorboard log
        tf_summary_files = glob.glob("*events.out.tfevents*")
        assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
        events_df = tensorboard_to_pandas(tf_summary_files[0])

        # compute maximum mean reward
        max_mean_reward = np.max(np.asarray(events_df.loc["train_BaseEnvEvents/reward/mean"]))
        return float(max_mean_reward)


if __name__ == "__main__":
    maze_run()
