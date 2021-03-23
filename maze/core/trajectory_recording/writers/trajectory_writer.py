"""Trajectory data is logged through trajectory writer objects registered via the global trajectory writer registry."""

from abc import ABC, abstractmethod

from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord


class TrajectoryWriter(ABC):
    """
    Interface for modules serializing the trajectory data.

    Implement this interface for any custom trajectory data logging.
    """

    @abstractmethod
    def write(self, episode_record: StateTrajectoryRecord) -> None:
        """
        Write out provided episode data (into a file, DB etc.)

        :param episode_record: Record of the episode trajectory data.
        """
