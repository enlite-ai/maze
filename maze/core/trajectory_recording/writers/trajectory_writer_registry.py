"""
Trajectory writers need to be globally registered in order to receive episode trajectory data streamed from
environments.
"""

from typing import List

from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter


class TrajectoryWriterRegistry:
    """
    Handles registration of trajectory data writers. Registered writers will
    be forwarded episode trajectory data at the end of each episode.
    """

    # List of registered writers
    writers: List[TrajectoryWriter] = []

    @classmethod
    def register_writer(cls, writer: TrajectoryWriter) -> None:
        """
        Register a writer. Each writer will receive all globally recorded trajectory data.

        :param writer: Trajectory writer to register.
        """
        cls.writers.append(writer)

    @classmethod
    def record_trajectory_data(cls, episode_record: StateTrajectoryRecord) -> None:
        """
        Record trajectory data through all registered trajectory data writers.

        :param episode_record: Record of episode trajectory data to log.
        """
        for writer in cls.writers:
            writer.write(episode_record)
