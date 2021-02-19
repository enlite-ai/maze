"""Contains a Bernoulli distribution."""
from typing import Sequence

import torch
from gym import spaces
from torch.distributions import Bernoulli, kl_divergence

from maze.core.annotations import override
from maze.distributions.torch_dist import TorchProbabilityDistribution


class BernoulliProbabilityDistribution(TorchProbabilityDistribution[Bernoulli]):
    """Bernoulli Torch probability distribution for multi-binary action spaces.

    :param logits: the action selection logits.
    :param action_space: The gym action space.
    :param temperature: The distribution temperature parameter.
    """

    @classmethod
    @override(TorchProbabilityDistribution)
    def required_logits_shape(cls, action_space: spaces.MultiBinary) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return [action_space.n]

    def __init__(self, logits: torch.Tensor, action_space: spaces.MultiBinary, temperature: float = 1.0):
        # apply temperature - can be applied in logits space in the same way as in the categorical distribution
        self.logits = logits / temperature
        super().__init__(dist=Bernoulli(logits=self.logits), action_space=action_space)

    @override(TorchProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        log_prob = super().log_prob(actions)
        assert self.dist.logits.shape == log_prob.shape
        return log_prob

    @override(TorchProbabilityDistribution)
    def deterministic_sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return self.dist.probs > 0.5

    @override(TorchProbabilityDistribution)
    def entropy(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return self.dist.entropy().mean(dim=-1)

    @override(TorchProbabilityDistribution)
    def kl(self, other: 'TorchProbabilityDistribution') -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return kl_divergence(self.dist, other.dist).mean(dim=-1)
