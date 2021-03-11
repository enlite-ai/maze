"""Composer interface for critic (value function) networks."""
from abc import abstractmethod, ABC
from typing import Any


class CriticComposerInterface(ABC):
    """Interface for critic (value function) network composers.
    """

    @property
    @abstractmethod
    def critic(self) -> Any:
        """value networks"""
