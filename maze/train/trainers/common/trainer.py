"""Contains an interface for trainers."""
import dataclasses
from abc import ABC, abstractmethod
from typing import Union, Optional, TypeVar, ClassVar

from typing.io import BinaryIO

from maze.core.agent.torch_model import TorchModel
from maze.train.trainers.common.config_classes import AlgorithmConfig


class Trainer(ABC):
    """
    Interface for trainers.
    :param algorithm_config: Algorithm configuration including all parameter expected in .train().
    :param model: Model to train.
    """

    AlgorithmConfigType: TypeVar = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
    _TorchModelType: TypeVar = TypeVar("_TorchModelType", bound=TorchModel)

    def __init__(self, algorithm_config: AlgorithmConfigType, model: Optional[TorchModel] = None):
        """
        Note: This is not implemented as dataclass due to type hinting for class members not working properly in derived
        classes with dataclasses. I.e. PyCharm's type hinting always assumes model is of type TorchModel, but not the
        specific subtype (e.g. TorchActorCritic).
        """

        self.model = model
        self.algorithm_config = algorithm_config

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
