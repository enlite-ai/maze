"""Abstract interface for imitation learning evaluation."""
import abc

from maze.core.agent.torch_policy import TorchPolicy


class ImitationEvaluator(abc.ABC):
    """Abstract interface for imitation learning evaluation."""

    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate
        """
