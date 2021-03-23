"""Simple default setup for event logging into CSV files."""

from typing import TypeVar

from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.log_stats.log_stats_writer_tensorboard import LogStatsWriterTensorboard
from maze.core.trajectory_recording.writers.trajectory_writer_file import TrajectoryWriterFile
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper

T = TypeVar("T")


class MonitoringSetup:
    """
    Simple setup for environment monitoring.

    Logs the following data:

      - Epoch statistics (console + Tensorboard format)
      - Environment events (TSV files, one per event type)
      - Trajectory data
    """

    def __init__(self, env: T, log_dir: str = '.'):
        """
        :param env: The environment to monitor
        :param log_dir: Where to log the monitoring data
        """
        self.env = env
        self.log_dir = log_dir

        # Wrap the env to enable stats, events, and trajectory data logging
        self.env = LogStatsWrapper.wrap(self.env, logging_prefix="eval")
        self.env = TrajectoryRecordingWrapper.wrap(self.env)

    def __enter__(self) -> T:
        """Register data writers."""
        # Register stats, events, and trajectory data writers
        print("*******", self.log_dir)
        register_log_stats_writer(LogStatsWriterConsole())
        register_log_stats_writer(LogStatsWriterTensorboard(log_dir=self.log_dir + "/stats",
                                                            tensorboard_render_figure=True))
        LogEventsWriterRegistry.register_writer(LogEventsWriterTSV(log_dir=self.log_dir + "/event_logs"))
        TrajectoryWriterRegistry.register_writer(TrajectoryWriterFile(log_dir=self.log_dir + "/trajectory_data"))
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Ensure data are written out in case the rollout ended in the middle of episode.
        """
        self.env.reset()
        self.env.write_epoch_stats()
