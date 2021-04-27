"""This file contains the MazeRllibSaveModelCallback which inherits from the tune callback system and is responsible
    for storing the models in a maze-compatible way, as well as cleaning and managing other important dump files."""
import os
import pickle
import shutil
from typing import List

import numpy as np
import torch
from ray.tune import Callback
from ray.tune.checkpoint_manager import QueueItem
from ray.tune.registry import RLLIB_MODEL, _global_registry
from ray.tune.trial import Trial


class MazeRLlibSaveModelCallback(Callback):
    """Tune Callbacks that manage the storing of the maze model, as well as delete duplicate and unnecessary files"""

    def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info) -> None:
        """Called after starting a trial instance.
            Here we clean up the directory and move all files (which are created by the main thread) to the the dir
            of the trial instance.

        :param iteration: Number of iterations of the tuning loop.
        :param trials: List of trials.
        :param trial: Trial that just has been started.
        :param info: Kwargs dict for forward compatibility.
        """

        # Copy .hydra directory to experiment subdirectory
        if os.path.exists(".hydra"):
            shutil.move(".hydra", trial.logdir)
        shutil.move("hydra_config.yaml", trial.logdir)

        # Copy the statistics file to the trial directory and delete the file if there is only one trial
        if 'wrappers' in trial.config['env_config']:
            if 'ObservationNormalizationWrapper' in trial.config['env_config']['wrappers']:
                statistics_file_name = \
                    trial.config['env_config']['wrappers']['ObservationNormalizationWrapper']['statistics_dump']

                statistics_path = os.path.abspath(os.path.join(trial.local_dir, os.path.pardir, statistics_file_name))
                new_statistics_path = os.path.join(trial.logdir, statistics_file_name)

                assert os.path.exists(statistics_path)
                assert not os.path.exists(new_statistics_path)

                shutil.copy(statistics_path, new_statistics_path)

                # Remove file if only one trial or if this is the last one
                if len(trials) == 1 or trial == trials[-1]:
                    os.remove(statistics_path)

    @staticmethod
    def delete_nan_checkpoints(trial: Trial, score_att: str) -> None:
        """Delete all entries from the checkpoint manager which have score_attribute value == nan

        :param trial: Trial that just saved a checkpoint.
        :param score_att: The score attribute
        """
        # Delete from best checkpoints

        best_checkpoints = trial.checkpoint_manager._best_checkpoints
        to_del = []
        for idx, checkpoint in enumerate(best_checkpoints):
            if isinstance(checkpoint, QueueItem):
                checkpoint = checkpoint.value
            if np.isnan(checkpoint.result[score_att]):
                to_del.append((idx, checkpoint))

        for idx, checkpoint in to_del[::-1]:
            # Delete from best_checkpoint list
            del trial.checkpoint_manager._best_checkpoints[idx]
            # Delete folder
            trial.checkpoint_manager.delete(checkpoint)
            # Delete from membership
            trial.checkpoint_manager._membership.remove(checkpoint)

    def on_trial_save(self, iteration: int, trials: List[Trial], trial: Trial, **info) -> None:
        """Called after receiving a checkpoint from a trial.

            In this method the model is saved as a maze compatible state dict in the new checkpoint. Furthermore, the
            best model from the checkpoints is saved in the logdir where the rest of the dumps are stored.

        Arguments:
            iteration (int): Number of iterations of the tuning loop.
            trials (List[Trial]): List of trials.
            trial (Trial): Trial that just saved a checkpoint.
            **info: Kwargs dict for forward compatibility.
        """

        if trial.config['model']['custom_model'] == 'maze_model':
            model_cls = _global_registry.get(RLLIB_MODEL, trial.config['model']['custom_model'])

            # Retrieve save name
            state_dict_dump_file = trial.config['model']['custom_model_config']['state_dict_dump_file']
            maze_save_path = trial.checkpoint.value + '_' + state_dict_dump_file

            # Load RLlib version of the model
            meta = pickle.load(open(trial.checkpoint.value, 'rb'))
            state_dict = pickle.loads(meta['worker'])['state']['default_policy']

            # Save maze model in checkpoint dir
            if not os.path.exists(maze_save_path):
                maze_state_dict = model_cls.get_maze_state_dict(state_dict)
                torch.save(maze_state_dict, maze_save_path)

            # Store best checkpoint model in logdir
            best_ist_smallest = trial.checkpoint_score_attr.startswith('min-')
            score_att = trial.checkpoint_score_attr.replace('min-', '') if best_ist_smallest \
                else trial.checkpoint_score_attr

            # Delete all checkpoints that are nan from the checkpoints manager. This is a bug in RLLIb!!!
            self.delete_nan_checkpoints(trial, score_att)

            # Get the best model of the checkpoints and store it in the logdir
            best_checkpoints = trial.checkpoint_manager._best_checkpoints
            best_overall_checkpoint = None
            if len(best_checkpoints) > 0 and best_checkpoints[-1].value is not None:
                best_overall_checkpoint = best_checkpoints[-1].value
            elif len(best_checkpoints) == 0 and best_ist_smallest:
                # Bug in RLLIB where min- values are not added to the best list for some reason
                best_overall_checkpoint = trial.checkpoint_manager.newest_persistent_checkpoint

            if best_overall_checkpoint is not None:
                best_checkpoint_maze_state_dict_path = best_overall_checkpoint.value + '_' + state_dict_dump_file
                if os.path.exists(best_checkpoint_maze_state_dict_path):
                    shutil.copy(best_checkpoint_maze_state_dict_path, os.path.join(trial.logdir, state_dict_dump_file))

            for file in os.listdir(os.path.join(trial.local_dir, os.path.pardir)):
                if file.endswith('.pdf'):
                    os.remove(file)
                if file.endswith('.pkl'):
                    os.remove(file)
