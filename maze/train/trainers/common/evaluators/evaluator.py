"""Abstract interface for policy evaluation."""

from __future__ import annotations

import abc

from maze.core.agent.torch_policy import TorchPolicy

# ruff: noqa: B027, B024


class Evaluator(abc.ABC):
    """Abstract interface for policy evaluation."""

    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy.

        For most use cases, this method is expected to:
          - Write the evaluation results into the statistic logs
          - Serialize the policy if the reward improved

        :param policy: Policy to evaluate
        """
