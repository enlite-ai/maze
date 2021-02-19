"""Contains an interface for trainers."""
from abc import ABC, abstractmethod
from typing import Union

from typing.io import BinaryIO


class Trainer(ABC):
    """ Interface for trainers. """

    @abstractmethod
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """Load state from file.
        This is required for resuming training or model fine tuning with different parameters.

        :param file_path: Path from where to load the state.
        """
