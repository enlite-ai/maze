"""Encapsulation of torch policy outputs for training  in structured environments."""

from __future__ import annotations

import dataclasses

from maze.core.env.action_conversion import TorchActionType
from maze.core.env.structured_env import ActorID
from maze.distributions.dict import DictProbabilityDistribution

import torch


@dataclasses.dataclass
class PolicySubStepOutput:
    """Dataclass for holding the output of the policy's compute full output method"""

    action_logits: dict[str, torch.Tensor]
    """A logits dictionary (action_head maps to action_logits) to parameterize the distribution from."""

    prob_dist: DictProbabilityDistribution
    """The respective instance of a DictProbabilityDistribution."""

    embedding_logits: dict[str, torch.Tensor] | None
    """The Embedding output if applicable, used as the input for the critic network."""

    actor_id: ActorID
    """The actor id of the output"""

    @property
    def entropy(self) -> torch.Tensor:
        """The entropy of the probability distribution."""
        return self.prob_dist.entropy()


class PolicyOutput:
    """A structured representation of a policy output over a full (flat) environment step."""

    def __init__(self):
        self._step_policy_outputs: list[PolicySubStepOutput] = []

    def __getitem__(self, item: int) -> PolicySubStepOutput:
        """Get a specified (by index) substep output"""
        return self._step_policy_outputs[item]

    def append(self, value: PolicySubStepOutput):
        """Append a given PolicySubStepOutput."""
        self._step_policy_outputs.append(value)

    def actor_ids(self) -> list[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return list(map(lambda x: x.actor_id, self._step_policy_outputs))

    @property
    def action_logits(self) -> list[dict[str, torch.Tensor]]:
        """List of action logits for the individual sub-steps"""
        return [po.action_logits for po in self._step_policy_outputs]

    @property
    def prob_dist(self) -> list[DictProbabilityDistribution]:
        """List of probability dictionaries for the individual sub-steps"""
        return [po.prob_dist for po in self._step_policy_outputs]

    @property
    def entropies(self) -> list[torch.Tensor]:
        """List of entropies (of the probability distribution of the individual sub-steps."""
        return [po.entropy for po in self._step_policy_outputs]

    @property
    def embedding_logits(self) -> list[dict[str, torch.Tensor]]:
        """List of embedding logits for the individual sub-steps"""
        return [po.embedding_logits for po in self._step_policy_outputs]

    def log_probs_for_actions(self, actions: list[TorchActionType]) -> list[TorchActionType]:
        """Compute the action log probs for given actions.
        :param actions: The actions to use.
        :return: The computed action log probabilities.
        """
        assert len(self.prob_dist) == len(actions)
        return [pb.log_prob(ac) for pb, ac in zip(self.prob_dist, actions, strict=False)]
