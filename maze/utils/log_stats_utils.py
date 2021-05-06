"""Utils to simplify the setup of the logging system.

Especially for demo snippets that require only basic console logging.
"""
import glob
import os
import pickle
import sys
from typing import Union

from omegaconf import DictConfig, OmegaConf

from maze.core.env.base_env import BaseEnv
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_stats.log_stats import GlobalLogState
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.log_stats.log_stats_writer_tensorboard import LogStatsWriterTensorboard
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.utils.bcolors import BColors


class SimpleStatsLoggingSetup:
    """
    Helper class to simplify the statistics logging setup. All statistics defined for the given env are sent
    to a console writer.

    Limitation: It can only handle a single environment.
    """

    def __init__(self, env: LogStatsEnv, log_dir: str = None):
        assert isinstance(env, LogStatsEnv) and isinstance(env, BaseEnv)
        self.env = env
        self.log_dir = log_dir

    def __enter__(self) -> None:
        """Register a episode statistics aggregator with the provided log stats environment."""

        # step logging setup: write to console
        register_log_stats_writer(LogStatsWriterConsole())
        if self.log_dir is not None:
            register_log_stats_writer(LogStatsWriterTensorboard(log_dir=self.log_dir, tensorboard_render_figure=True))

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure that the statistics of the current log step is calculated and written to the console """
        # we need to trigger the episode statistics calculation if we stopped the simulation before it was done
        self.env.reset()

        # calculate the epoch statistics, based on the individual episode statistics produced in the loop above.
        self.env.write_epoch_stats()


def clear_global_state():
    """Resets the seed and global state to ensure that consecutive tests run under the same preconditions."""
    # ensure that there are no left-overs from previous runs
    LogEventsWriterRegistry.writers = []  # clear writers
    TrajectoryWriterRegistry.writers = []  # clear writers

    for writer in GlobalLogState.global_log_stats_writers:
        writer.close()

    GlobalLogState.global_step = 1
    GlobalLogState.hook_on_log_step = []
    GlobalLogState.global_log_stats_writers = []


def setup_logging(job_config: Union[DictConfig, str, None], log_dir: str = ".") -> None:
    """Setup tensorboard logging, derive the logging directory from the script name.

    :param job_config: Configuration written as text to tensorboard (experiment config).
    :param log_dir: log_dir for TensorBoard.
    """
    # hydra handles the working directory
    writer = LogStatsWriterTensorboard(log_dir=log_dir, tensorboard_render_figure=True)
    register_log_stats_writer(writer)
    # attach a console writer as well for immediate console feedback
    register_log_stats_writer(LogStatsWriterConsole())

    summary_writer = writer.summary_writer
    summary_writer.add_text("cmd", " ".join(sys.argv))

    if job_config is not None:
        # log run settings
        if isinstance(job_config, DictConfig):
            if job_config.__dict__["_metadata"].flags.get("allow_objects", False):
                # Hydra was instructed to allow objects. This was done by the Python training API, hence we might have
                # Python objects in our config, which makes logging the config to file not possible.
                # todo Making this work for the Python training API would require reversing the Hydra instantiaton, i.e.
                #  generate a Hydra configuration from Python objects (at least partially).
                BColors.print_colored(
                    "Logging run configurations with injected non-primitives is not supported yet. For now please don't"                    
                    " inject non-primitives if you wish to log the configuration of your run.",
                    BColors.WARNING
                )
                return
            else:
                job_config = OmegaConf.to_yaml(job_config)

        # prepare config text for tensorboard
        job_config = job_config.replace("\n", "</br>")
        job_config = job_config.replace(" ", "&nbsp;")

        summary_writer.add_text("job_config", job_config)

    # Load the figures from the given files and add them to tensorboard.
    for net_image_path in glob.glob('*.figure.pkl'):
        network_description = net_image_path.split('/')[-1].replace('.figure.pkl', '')
        fig = pickle.load(open(net_image_path, 'rb'))
        summary_writer.add_figure(f'{network_description}', fig, close=True)
        os.remove(net_image_path)
