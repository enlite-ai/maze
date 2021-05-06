"""Implements the Maze command line interface for running rollouts, trainings and else."""
import glob
import os
from typing import Optional

import hydra
import matplotlib
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from maze.core.log_stats.hparam_writer_tensorboard import manipulate_hparams_logging_for_exp
from maze.core.utils.factory import Factory
from maze.core.utils.seeding import MazeSeeding
from maze.runner import Runner
from maze.utils.bcolors import BColors
from maze.utils.log_stats_utils import clear_global_state
from maze.utils.tensorboard_reader import tensorboard_to_pandas


def set_matplotlib_backend() -> None:
    """Switch matplotlib backend for maze runs on headless machines to Agg (non-interactive).
    """
    if not os.environ.get('MPLBACKEND') and not os.environ.get('DISPLAY'):
        BColors.print_colored(f"INFO: No display detected! Switching matplotlib to headless backend Agg!",
                              color=BColors.OKBLUE)
        matplotlib.use('Agg')


def _run_job(cfg: DictConfig) -> None:
    """Runs a regular maze job.

    :param cfg: Hydra configuration for the rollout.
    """
    set_matplotlib_backend()

    # If no env or agent base seed is given generate the seeds randomly and add them to the resolved hydra config
    if cfg.seeding.env_base_seed is None:
        cfg.seeding.env_base_seed = MazeSeeding.generate_seed_from_random_state(np.random.RandomState(None))
    if cfg.seeding.agent_base_seed is None:
        cfg.seeding.agent_base_seed = MazeSeeding.generate_seed_from_random_state(np.random.RandomState(None))

    # print and log config
    config_str = yaml.dump(OmegaConf.to_container(cfg, resolve=True))
    with open("hydra_config.yaml", "w") as fp:
        fp.write("\n" + config_str)
    BColors.print_colored(config_str, color=BColors.HEADER)
    print("Output directory: {}\n".format(os.path.abspath(".")))

    # run job
    runner = Factory(base_type=Runner).instantiate(cfg.runner)
    runner.setup(cfg)
    runner.run()


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

    # Add hparams logging to tensorboard
    metrics = [('train_BaseEnvEvents/reward/mean', max_mean_reward, 'max')]
    manipulate_hparams_logging_for_exp('.', metrics, clear_hparams=False)

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
    is_multi_run = instance.cfg is not None and instance.cfg.hydra.job.get("num") is not None

    # regular single runs
    if not is_multi_run:
        _run_job(cfg)
    # multirun (e.g., gird search, nevergrad, ...)
    else:
        max_mean_reward = _run_multirun_job(cfg)
        return max_mean_reward


if __name__ == "__main__":
    maze_run()
