"""Extends the probability distribution interface for PyTorch distributions."""
from abc import ABC, abstractmethod
from typing import Sequence, Generic, TypeVar

import torch
import torch.distributions as torch_dist
from gym import spaces
from torch.distributions import Categorical

from maze.core.annotations import override
from maze.distributions.distribution import ProbabilityDistribution

T = TypeVar("T", bound=torch_dist.Distribution)


class TorchProbabilityDistribution(ProbabilityDistribution, Generic[T], ABC):
    """Base class for wrapping Torch probability distributions.

    :param dist: The torch probability distribution.
    :param action_space: The gym action space.
    """

    @classmethod
    @abstractmethod
    def required_logits_shape(cls, action_space: spaces.Space) -> Sequence[int]:
        """Returns the required shape for the corresponding neural network logits output.

        :param action_space: The respective action space to compute logits for.
        :return: The required logits shape.
        """

    def __init__(self, dist: T, action_space: spaces.Space):
        self.dist = dist
        self.action_space = action_space

    @override(ProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        log_prob = self.dist.log_prob(actions)
        return log_prob

    @override(ProbabilityDistribution)
    def entropy(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return self.dist.entropy()

    @override(ProbabilityDistribution)
    def kl(self, other: 'TorchProbabilityDistribution') -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return torch_dist.kl.kl_divergence(self.dist, other.dist)

    @override(ProbabilityDistribution)
    def sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return self.dist.sample()
