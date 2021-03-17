"""An imitation dataset that loads all data to memory sequentially."""
import logging
from pathlib import Path
from typing import Union

from maze.core.annotations import override
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset

logger = logging.getLogger(__name__)


class SequentialLoadDataset(InMemoryDataset):
    """A version of the in-memory dataset that loads all data sequentially.

    Useful when conversion of data on load (i.e. maze states into observations etc.) is either fast or not required.
    """

    @override(InMemoryDataset)
    def load_data(self, dir_or_file: Union[str, Path]) -> None:
        """Load the trajectory data based on arguments provided on init."""
        logger.info(f"Started loading trajectory data from: {dir_or_file}")

        for trajectory in self.deserialize_trajectories(dir_or_file):
            self.append(trajectory)

        logger.info(f"Loaded trajectory data from: {dir_or_file}")
        logger.info(f"Current length is {len(self)} steps in total.")
