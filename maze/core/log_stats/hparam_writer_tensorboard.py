"""File containing methods for adding hparams logging in tensorboard for a given experiment directory"""
import glob
import os
from typing import List, Union, Dict, Any, Callable, Tuple

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from maze.utils.bcolors import BColors
from maze.utils.tensorboard_reader import tensorboard_to_pandas


def flatten(struc_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Union[str, int, float]]:
    """Flatten the given dict such that keys are concatenated by the given sep parameter

    :param struc_dict: The structured dictionary to be flattened.
    :param parent_key: The key of the parent dict value.
    :param sep: The separator to use for concatenating the keys.

    :return: The flattened dict.
    """
    items = []
    for k, v in struc_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(list(v))))
        elif isinstance(v, (str, float, int)):
            items.append((new_key, v))
        elif v is None:
            items.append((new_key, v))
        else:
            raise ValueError(f'Unexpected value type found: {type(v), v}')
    return dict(items)


def manipulate_hparams_logging_for_exp(exp_dir: str, metrics: List[Tuple[str, Union[Callable[[np.ndarray],
                                                                                             float], float], str]],
                                       clear_hparams: bool) -> None:
    """Manipulate the hparams logging for a given experiment directory.

    That is ether add hparams logging by adding a new tfevents file, or replace an already present hparams event files,
    or simple remove all hparams event files.

    :param exp_dir: The experiment directory.
    :param metrics: A list of metrics to be added to tensorboard. Each tuple in the list should consist of the key to
        query the original events file, the funciton to aggregate the values to a single float values and a simple
        name describing the function.. e.g. metrics=[('train_BaseEnvEvents/reward/mean', np.max, 'max')]
        Note: Instead of the callable a float value can be given as well, which will be used instead of querying the
        file.
    :param clear_hparams: Optional value specifying if the hparams file should simple be deleted.
    """

    # Get hparams file if present and delete it
    tf_hparams_summary_files = glob.glob(f"{exp_dir}/*events.out.tfevents*_hparams")
    if len(tf_hparams_summary_files) > 0:
        for ff in tf_hparams_summary_files:
            os.remove(ff)

    # Assert that only one events file is present
    tf_summary_files = glob.glob(f"{exp_dir}/*events.out.tfevents*")
    hydra_config_file = os.path.join(exp_dir, '.hydra/config.yaml')
    if len(tf_summary_files) == 0 or not os.path.exists(hydra_config_file):
        return
    assert len(tf_summary_files) == 1

    if not clear_hparams:

        # Read confg.yaml file as hyperparameters
        assert os.path.exists(hydra_config_file)
        cfg = yaml.safe_load(open(hydra_config_file))
        hparam_dict = flatten(dict(cfg))

        # compute maximum for each given metric from the original events file
        metrics_dict = dict()
        for (metric_key, metric_func, metric_func_name) in metrics:
            try:
                if isinstance(metric_func, float):
                    new_metric_name = f'{metric_key}-{metric_func_name}'
                    metrics_dict[new_metric_name] = metric_func
                else:
                    events_df = tensorboard_to_pandas(tf_summary_files[0])
                    new_metric_name = f'{metric_key}-{metric_func_name}'
                    metrics_dict[new_metric_name] = metric_func(np.asarray(events_df.loc[metric_key]))
            except KeyError:
                BColors.print_colored(
                    f'The given metric key: {metric_key} could not be found in the summary file for exp: '
                    f'{exp_dir}', BColors.WARNING)

        # Store all files and dirs present in the directory before creating a new summary file writer
        all_elems_in_exp_dir_before = set(os.listdir(exp_dir))

        # Create the summary file writer
        summary_writer = SummaryWriter(log_dir=exp_dir, filename_suffix='_hparams')

        # Add hparams to the summary writer and close
        print(f'- Adding Tensorflow hparams events for runs in directory: {exp_dir}')
        summary_writer.add_hparams(hparam_dict, metrics_dict)
        summary_writer.close()

        # Check what new files and dirs have been created.
        new_elems = set(os.listdir(exp_dir)) - all_elems_in_exp_dir_before
        new_dirs = list(filter(lambda x: os.path.isdir(os.path.join(exp_dir, x)), list(new_elems)))
        new_files = list(filter(lambda x: os.path.isfile(os.path.join(exp_dir, x)), list(new_elems)))

        # There should be one new file and one new dir
        assert len(new_dirs) == 1, new_dirs
        assert len(new_files) == 1, new_files

        # Get the proper hparams events file
        proper_file = os.listdir(os.path.join(exp_dir, new_dirs[0]))
        assert len(proper_file) == 1

        # Remove emtpy file (created from some unknown reason)
        os.remove(os.path.join(exp_dir, new_files[0]))

        # Move the proper hparams events file into the same dir, for same naming in tensorboard
        os.rename(os.path.join(exp_dir, new_dirs[0], proper_file[0]), os.path.join(exp_dir, new_files[0]))

        # Remove the now empty dir created by the summary writer
        os.rmdir(os.path.join(exp_dir, new_dirs[0]))
