"""Contains performance unit tests."""
import glob
import os
from typing import Dict

import numpy as np

import pytest
from hydra.experimental import compose, initialize_config_module
from maze.maze_cli import maze_run
from maze.utils.tensorboard_reader import tensorboard_to_pandas
from maze.utils.timeout import Timeout

# Configurations to be tested
trainings = [
    # PPO
    [180, {"algorithm": "ppo", "algorithm.n_epochs": "2", "algorithm.eval_repeats": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # A2C
    [180, {"algorithm": "a2c", "algorithm.n_epochs": "3", "algorithm.eval_repeats": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # IMPALA
    [180, {"algorithm": "impala", "algorithm.n_epochs": "4", "algorithm.eval_repeats": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # ES
]


@pytest.mark.longrun
@pytest.mark.parametrize("target_reward, hydra_overrides", trainings)
def test_train(hydra_overrides: Dict[str, str], target_reward: float, tmpdir: str):
    # set working directory to temp path
    os.chdir(tmpdir)

    with initialize_config_module(config_module="maze.conf"):

        # config is relative to a module
        cfg = compose(config_name="conf_train", overrides=[key + "=" + value for key, value in hydra_overrides.items()])

        # run training
        with Timeout(seconds=300):
            maze_run(cfg)

        # load tensorboard log
        tf_summary_files = glob.glob("*events.out.tfevents*")
        assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
        events_df = tensorboard_to_pandas(tf_summary_files[0])

        # check if target reward was reached
        max_mean_reward = np.asarray(events_df.loc["train_BaseEnvEvents/reward/mean"]).max()
        assert max_mean_reward >= target_reward
