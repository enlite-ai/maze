"""Implements the Maze command line interface for running rollouts, trainings and else."""
import glob
from typing import Optional

import hydra
import matplotlib
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from maze.core.utils.factory import Factory
from maze.runner import Runner
from maze.utils.bcolors import BColors
from maze.utils.log_stats_utils import clear_global_state
from maze.utils.tensorboard_reader import tensorboard_to_pandas


def _run_job(cfg: DictConfig) -> None:
    """Runs a regular maze job.

    :param cfg: Hydra configuration for the rollout.
    """
    # switch matplotlib backend for maze runs (non-interactive)
    matplotlib.use('Agg')

    # print and log config
    config_str = yaml.dump(OmegaConf.to_container(cfg, resolve=True))
    with open("hydra_config.yaml", "w") as fp:
        fp.write(config_str)
    BColors.print_colored("\n" + config_str, color=BColors.HEADER)

    # run job
    runner = Factory(base_type=Runner).instantiate(cfg.runner)
    runner.run(cfg)


def _run_multirun_job(cfg: DictConfig) -> float:
    """Runs a maze job which is part of a Hydra --multirun and returns the maximum mean reward of this run.

    :param cfg: Hydra configuration for the rollout.
    :return: The maximum mean reward achieved throughout the training run.
    """
    # required for Hydra --multirun (e.g., sweeper)
    clear_global_state()

    try:
        _run_job(cfg)
    # when optimizing hyper parameters a single exception
    # in one job should not break the entire experiment
    except:
        return float(np.finfo(np.float32).min)

    # load tensorboard log and return maximum mean reward
    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df = tensorboard_to_pandas(tf_summary_files[0])

    # compute maximum mean reward
    max_mean_reward = np.max(np.asarray(events_df.loc["train_BaseEnvEvents/reward/mean"]))
    return float(max_mean_reward)


@hydra.main(config_path="conf", config_name="conf_rollout")
def maze_run(cfg: DictConfig) -> Optional[float]:
    """
    Run a CLI task based on the provided configuration.

    A runner object is instantiated according to the config (cfg.runner) and it is then handed
    the whole configuration object (cfg). Runners can perform various tasks such as rollouts, trainings etc.

    :param cfg: Hydra configuration for the rollout.
    :return: In case of a multirun it returns the maximum mean reward (required for hyper parameter optimization).
             For regular runs nothing is returned.
    """

    # check if we are currently in a --multirun
    instance = HydraConfig.instance()
    is_multi_run = instance.cfg.hydra.job.get("num") is not None

    # regular single runs
    if not is_multi_run:
        _run_job(cfg)
    # multirun (e.g., gird search, nevergrad, ...)
    else:
        max_mean_reward = _run_multirun_job(cfg)
        return max_mean_reward


if __name__ == "__main__":
    maze_run()
