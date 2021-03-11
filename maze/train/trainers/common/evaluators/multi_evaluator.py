"""Evaluating a policy using multiple evaluators in sequence."""

from typing import List

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.train.trainers.common.evaluators.evaluator import Evaluator


class MultiEvaluator(Evaluator):
    """Evaluates the given policy using multiple different evaluators (ran in sequence).

    Useful when evaluating a policy in different scenarios. E.g., during behavioral cloning,
    we might want to evaluate the policy first on a validation dataset and then through an evaluation rollout.

    :param evaluators: Evaluators to run.
    """

    def __init__(self, evaluators: List[Evaluator]):
        self.evaluators = evaluators

    @override(Evaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy using the given evaluators.

        :param policy: Policy to evaluate
        """
        for evaluator in self.evaluators:
            evaluator.evaluate(policy)
