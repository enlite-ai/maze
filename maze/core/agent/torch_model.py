"""Contains a base class for torch models."""
from abc import abstractmethod
from typing import List, Dict

import torch


class TorchModel:
    """Base class for any torch model.

    :param device: Device the networks should be located on (cpu or cuda)
    """

    def __init__(self, device: str):
        self._device = device
        self.to(self._device)

    @property
    def num_params(self) -> int:
        """Returns overall number of network parameters."""
        return sum(t.numel() for t in self.parameters())

    @property
    def device(self) -> str:
        """Returns the device the networks are located on."""
        return self._device

    @abstractmethod
    def parameters(self) -> List[torch.Tensor]:
        """Returns all parameters of all networks."""

    @abstractmethod
    def eval(self) -> None:
        """Set all networks to eval mode."""

    @abstractmethod
    def train(self) -> None:
        """Set all networks to training mode."""

    @abstractmethod
    def to(self, device: str) -> None:
        """Move all networks to the specified device.
        :param device: The target device.
        """

    @abstractmethod
    def state_dict(self) -> Dict:
        """Return state dict composed of state dicts of all encapsulated networks."""

    @abstractmethod
    def load_state_dict(self, state_dict: Dict) -> None:
        """Set state dict of all encapsulated networks.
        :param state_dict: The torch state dictionary.
        """
