"""Composer interface for critic (value function) networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CriticComposerInterface(ABC):
    """Interface for critic (value function) network composers."""

    @property
    @abstractmethod
    def critic(self) -> Any:
        """value networks"""
