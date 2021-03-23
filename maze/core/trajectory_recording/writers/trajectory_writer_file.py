"""Simple serialization of trajectory data using Pickle."""

import pickle
from pathlib import Path
from typing import Union

from maze.core.annotations import override
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter


class TrajectoryWriterFile(TrajectoryWriter):
    """
    Simple trajectory data writer. Serializes trajectory data for each episode
    into a separate file using Pickle.

    Suitable for smaller scale rollouts or debugging.

    :param log_dir: Where trajectory data should be logged.
    """

    def __init__(self, log_dir: Union[str, Path] = Path("./trajectory_data")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @override(TrajectoryWriter)
    def write(self, episode_record: StateTrajectoryRecord) -> None:
        """
        Write episode trajectory data to a file using pickle.

        :param episode_record: Episode trajectory data
        """
        filename = episode_record.id + ".pkl"
        with open(self.log_dir / filename, "wb") as out_f:
            pickle.dump(episode_record, out_f)
