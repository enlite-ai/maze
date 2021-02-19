from abc import ABC


class ModelSelectionBase(ABC):
    """Base class for model selection strategies."""

    def update(self, reward: float) -> None:
        """Receives a new evaluation result from the model.

        :param reward: mean evaluation reward
        """
