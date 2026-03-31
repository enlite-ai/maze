from __future__ import annotations

from abc import ABC

# ruff: noqa: B027, B024


class ModelSelectionBase(ABC):
    """Base class for model selection strategies."""

    def update(self, reward: float) -> None:
        """Receives a new evaluation result from the model. Should be only called once per epoch.

        :param reward: mean evaluation reward.
        """
