import os
import pickle
import shutil
from typing import Dict

import pytest
import torch
from omegaconf import DictConfig

from maze.rllib.maze_rllib_models.maze_rllib_ac_model import MazeRLlibACModel
from maze.test.shared_test_utils.hydra_helper_functions import run_maze_from_str


def run_cartpole(hydra_args: Dict[str, str]):
    """Test full run with cartpole"""
    hydra_overrides = {'rllib/runner': 'dev', 'configuration': 'test',
                       'env': 'gym_env', 'model': 'vector_obs', 'wrappers': 'vector_obs', 'critic': 'template_state',
                       **hydra_args}

    cfg = run_maze_from_str('maze.conf', 'conf_rllib', hydra_overrides)
    return cfg


@pytest.mark.rllib
def test_stored_model_multiple_check_points_max_att():
    check_stored_model_multiple_check_points({'runner.num_workers': '1',
                                              'runner.ray_config.local_mode': 'False',
                                              'runner.tune_config.keep_checkpoints_num': '2',
                                              'runner.ray_config.ignore_reinit_error': 'True',
                                              'runner.tune_config.stop.training_iteration': '2'})


@pytest.mark.rllib
def test_stored_model_multiple_check_points_min_att():
    check_stored_model_multiple_check_points({'runner.tune_config.keep_checkpoints_num': '2',
                                              'runner.tune_config.stop.training_iteration': '2',
                                              'runner.ray_config.ignore_reinit_error': 'True',
                                              'runner.tune_config.checkpoint_score_attr': 'min-episode_reward_max'})


def check_stored_model_multiple_check_points(hydra_args: Dict[str, str]):
    """Run the exp and check the output"""
    cfg = run_cartpole(hydra_args)

    assert os.path.exists(cfg.algorithm.algorithm)
    trial_dir = None
    for root, dirs, files in os.walk(cfg.algorithm.algorithm):
        if root == cfg.algorithm.algorithm:
            assert len(dirs) == 1, 'Should contain one directory for the trial'
            trial_dir = dirs[0]
            assert len(files) == 2, 'Should contain two files: experiment_state-date_time.json, and ' \
                                    'basic-variant-state-date_time.json'
        else:
            continue

    base_dir = os.path.join(cfg.algorithm.algorithm, trial_dir)
    for root, dirs, files in os.walk(base_dir):
        if base_dir == root:
            dirs.remove('DEPRECATED_VALUE')
            assert len(dirs) == cfg.runner.tune_config.keep_checkpoints_num, dirs
            for dir in dirs:
                assert 'checkpoint_' in dir

            assert cfg.runner.spaces_config_dump_file in files

            try:
                import pygraphviz
                assert len(list(filter(lambda x: x.endswith('.pdf'), files))) > 0
            except ImportError:
                pass

            if 'ObservationNormalizationWrapper' in cfg.wrappers:
                assert cfg.wrappers.ObservationNormalizationWrapper.statistics_dump in files

            # Since only one trial is started the statistics file should be deleted
            assert cfg.wrappers["maze.core.wrappers.observation_normalization.observation_normalization_wrapper.ObservationNormalizationWrapper"].statistics_dump not in os.listdir('.')

            for checkpoint_dir in dirs:
                check_checkpoint_dir(os.path.join(root, checkpoint_dir), checkpoint_dir, cfg)

            assert cfg['runner']['state_dict_dump_file'] in files

    shutil.rmtree(cfg.algorithm.algorithm)


def check_checkpoint_dir(checkpoint_path: str, checkpoint_name: str, cfg: DictConfig):
    """Check the directory structure of checkpoints"""

    checkpoint_name, checkpoint_id = checkpoint_name.split("_")
    checkpoint_name = checkpoint_name + "-" + checkpoint_id.lstrip("0")
    assert len(os.listdir(checkpoint_path)) == 4
    assert '.is_checkpoint' in os.listdir(checkpoint_path)
    assert checkpoint_name in os.listdir(checkpoint_path)
    assert checkpoint_name + '.tune_metadata' in os.listdir(checkpoint_path)

    maze_state_dict_name = checkpoint_name + '_' + cfg['runner']['state_dict_dump_file']
    assert maze_state_dict_name in os.listdir(checkpoint_path)

    ray_meta = pickle.load(open(os.path.join(checkpoint_path, checkpoint_name), 'rb'))
    state = pickle.loads(ray_meta['worker'])['state']
    ray_state_dict = state['default_policy']
    reconstructed_maze_state_dict = MazeRLlibACModel.get_maze_state_dict(ray_state_dict)
    loaded_maze_state_dict = torch.load(os.path.join(checkpoint_path, maze_state_dict_name))

    assert list(reconstructed_maze_state_dict.keys()) == list(loaded_maze_state_dict.keys())
    for key in reconstructed_maze_state_dict.keys():
        assert list(reconstructed_maze_state_dict[key].keys()) == list(loaded_maze_state_dict[key].keys())
        for step_key in reconstructed_maze_state_dict[key].keys():
            assert list(reconstructed_maze_state_dict[key][step_key].keys()) == \
                   list(loaded_maze_state_dict[key][step_key].keys())
            for weight_key in reconstructed_maze_state_dict[key][step_key].keys():
                assert torch.allclose(loaded_maze_state_dict[key][step_key][weight_key],
                                      reconstructed_maze_state_dict[key][step_key][weight_key]), \
                    f'{loaded_maze_state_dict[key]},\n\n {reconstructed_maze_state_dict[key]}'
