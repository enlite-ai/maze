"""Contains a diagonal Gaussian distributions"""
from typing import Sequence

import torch
from gym import spaces
from torch.distributions import Normal, kl_divergence

from maze.core.annotations import override
from maze.distributions.torch_dist import TorchProbabilityDistribution


class DiagonalGaussianProbabilityDistribution(TorchProbabilityDistribution[Normal]):
    """Diagonal Gaussian (Normal) Torch probability distribution.

    :param logits: The logits for both mean and standard deviation.
    """

    @classmethod
    @override(TorchProbabilityDistribution)
    def required_logits_shape(cls, action_space: spaces.Space) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        required_shape = [2 * v for v in action_space.shape]
        return required_shape

    def __init__(self, logits: torch.Tensor, action_space: spaces.Box, temperature: float):
        # split distribution parameters
        mean, log_std = torch.chunk(logits, chunks=2, dim=-1)

        # apply temperature
        log_std = log_std / temperature

        super().__init__(dist=Normal(mean, torch.exp(log_std)), action_space=action_space)

    @override(TorchProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        log_prob = self.dist.log_prob(actions).mean(-1)
        assert self.dist.mean.shape[:-1] == log_prob.shape
        return log_prob

    @override(TorchProbabilityDistribution)
    def deterministic_sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return self.dist.mean

    @override(TorchProbabilityDistribution)
    def entropy(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return self.dist.entropy().mean(dim=-1)

    @override(TorchProbabilityDistribution)
    def kl(self, other: 'TorchProbabilityDistribution') -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return kl_divergence(self.dist, other.dist).mean(dim=-1)
