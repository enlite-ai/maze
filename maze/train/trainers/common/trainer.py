"""Contains an interface for trainers."""
from abc import ABC, abstractmethod
from typing import Union, Optional, TypeVar

from typing.io import BinaryIO

from maze.core.agent.torch_model import TorchModel
from maze.train.trainers.common.config_classes import AlgorithmConfig


class Trainer(ABC):
    """
    Interface for trainers.
    :param config: Algorithm configuration including all parameter expected in .train().
    """

    _AlgorithmConfigType = TypeVar("_AlgorithmConfigType", bound=AlgorithmConfig)

    def __init__(self, config: _AlgorithmConfigType):
        self.model: Optional[TorchModel] = None
        self.algorithm_config: Trainer._AlgorithmConfigType = config

    @abstractmethod
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """Load state from file.
        This is required for resuming training or model fine tuning with different parameters.

        :param file_path: Path from where to load the state.
        """

    @abstractmethod
    def train(self, n_epochs: Optional[int] = None, **kwargs) -> None:
        """
        Train for n epochs. kwargs describe additional configuration necessary at training time, as for e.g. ESTrainer
        or BCTrainer.
        Some necessary parameters are set at initialization time and can be read from the trainer's algorithm config.
        Hence all parameters available at initialization time are optional to set at train time.
        :param n_epochs: Number of epochs to train.
        :param kwargs: Additional, trainer-specific parameters.
        """
