"""Test the seeding of the algorithm"""

# Configurations to be tested
import glob
import os
from typing import Dict

import pytest

from maze.test.shared_test_utils.run_maze_utils import run_maze_job
from maze.utils.log_stats_utils import clear_global_state
from maze.utils.tensorboard_reader import tensorboard_to_pandas
from maze.utils.timeout import Timeout

trainings = [
    {"algorithm": "ppo", "configuration": "test",
     "env": "gym_env", "env.name": "CartPole-v0"},

    {"algorithm": "a2c", "configuration": "test",
     "env": "gym_env", "env.name": "CartPole-v0"},

    {"algorithm": "es", "configuration": "test",
     "env": "gym_env", "env.name": "CartPole-v0"}
]


@pytest.mark.parametrize("hydra_overrides", trainings)
def test_algorithm_seeding_trainings(hydra_overrides: Dict[str, str]):
    perform_algorithm_seeding_test(hydra_overrides)


def perform_algorithm_seeding_test(hydra_overrides: Dict[str, str]):
    # Perform base run for comparison ----------------------------------------------------------------------------------
    base_dir = os.path.abspath('.')
    os.mkdir('./base_exp')
    os.chdir('./base_exp')
    # run training
    with Timeout(seconds=60):
        cfg = run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_train")

    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df = tensorboard_to_pandas(tf_summary_files[0])
    clear_global_state()

    # Perform comparison run with same seeds ---------------------------------------------------------------------------
    os.chdir(base_dir)
    os.mkdir('./exp_pos')
    os.chdir('./exp_pos')

    hydra_overrides['seeding.agent_base_seed'] = cfg.seeding.agent_base_seed
    hydra_overrides['seeding.env_base_seed'] = cfg.seeding.env_base_seed
    # run training
    with Timeout(seconds=60):
        run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_train")

    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df_2 = tensorboard_to_pandas(tf_summary_files[0])
    clear_global_state()
    del hydra_overrides['seeding.agent_base_seed']
    del hydra_overrides['seeding.env_base_seed']

    assert len(events_df) == len(events_df_2)
    for idx, (key, epoch) in enumerate(events_df.index):
        if 'time' in key:
            continue
        assert events_df_2.values[idx] == events_df.values[idx], \
            f'Value not equal for key: {key} in epoch: {epoch}'

    # Perform second comparison run with different seeds ---------------------------------------------------------------
    os.chdir(base_dir)
    os.mkdir('./exp_neg')
    os.chdir('./exp_neg')

    # run training
    with Timeout(seconds=60):
        run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_train")

    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df_2 = tensorboard_to_pandas(tf_summary_files[0])

    all_equal = True
    for idx, (key, epoch) in enumerate(events_df.index):
        if 'time' in key:
            continue
        all_equal = all_equal and events_df.values[idx] == events_df_2.values[idx]
    assert not all_equal, 'The resulting logs should not be all equal'
