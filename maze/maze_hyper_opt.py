"""Implements the Maze command line interface for running rollouts, trainings and else."""
import glob

import hydra
import numpy as np
from omegaconf import DictConfig

from maze.maze_cli import maze_run
from maze.utils.log_stats_utils import clear_global_state
from maze.utils.tensorboard_reader import tensorboard_to_pandas


@hydra.main(config_path="conf", config_name="conf_rollout")
def maze_hyper_opt(cfg: DictConfig) -> float:
    """
    Run a CLI task based on the provided configuration.

    A runner object is instantiated according to the config (cfg.runner) and it is then handed
    the whole configuration object (cfg). Runners can perform various tasks such as rollouts, trainings etc.

    :param cfg: Hydra configuration for the rollout.
    """
    # required for Hydra --multirun (e.g., sweeper)
    clear_global_state()

    # run original maze cli
    try:
        maze_run(cfg)
    except:
        # todo: unknown error
        return np.finfo(np.float32).min

    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    # todo: how is this possible?
    # assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df = tensorboard_to_pandas(tf_summary_files[0])

    # return maximum mean reward
    max_mean_reward = np.max(np.asarray(events_df.loc["train_BaseEnvEvents/reward/mean"]))
    return float(max_mean_reward)


if __name__ == "__main__":
    maze_hyper_opt()
