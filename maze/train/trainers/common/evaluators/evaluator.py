"""Abstract interface for policy evaluation."""
import abc

from maze.core.agent.torch_policy import TorchPolicy


class Evaluator(abc.ABC):
    """Abstract interface for policy evaluation."""

    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy.

        For most use cases, this method is expected to:
          - Write the evaluation results into the statistic logs
          - Serialize the policy if the reward improved

        :param policy: Policy to evaluate
        """
