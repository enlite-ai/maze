"""Simple setup for trajectory data recording into the current directory."""

from maze.core.trajectory_recording.writers.trajectory_writer_file import TrajectoryWriterFile
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper


class SimpleTrajectoryRecordingSetup:
    """
    Simple setup for trajectory data recording.

    Trajectory data will be serialized into `trajectory_data` directory, one file per episode.
    """

    def __init__(self, env):
        assert isinstance(env, TrajectoryRecordingWrapper)
        self.env = env

    def __enter__(self) -> None:
        """Register trajectory data file writer."""
        TrajectoryWriterRegistry.register_writer(TrajectoryWriterFile())

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Ensure the trajectory data are written out in case the rollout ended in the middle of episode
        (without env reset at the end).
        """
        self.env.reset()
