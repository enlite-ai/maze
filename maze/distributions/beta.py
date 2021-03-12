"""Contains a beta distribution"""
import math
from typing import Sequence

import numpy as np
import gym
import torch
from gym import spaces
from torch.distributions import Beta, kl_divergence

from maze.core.annotations import override
from maze.distributions.torch_dist import TorchProbabilityDistribution
from maze.distributions.utils import EPSILON
from maze.perception.perception_utils import convert_to_torch


class BetaProbabilityDistribution(TorchProbabilityDistribution[Beta]):
    """Beta Torch probability distribution.

    :param logits: the logits for both mean and standard deviation.
    :param action_space: the underlying gym.spaces action space.
    """

    @classmethod
    @override(TorchProbabilityDistribution)
    def required_logits_shape(cls, action_space: spaces.Space) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        required_shape = [2 * v for v in action_space.shape]
        return required_shape

    def __init__(self, logits: torch.Tensor, action_space: gym.spaces.Box, temperature: float):

        # make sure space bounds are set properly
        assert np.all(action_space.low > -np.inf) and np.all(action_space.low < np.inf)

        # prepare logits for distribution
        logits = torch.clamp(logits, math.log(EPSILON), -math.log(EPSILON))
        logits = torch.log(torch.exp(logits) + 1.0) + 1.0

        # apply temperature to alpha, beta
        logits = logits / temperature

        # create low and high tensors for vectorized clamping
        self.low = convert_to_torch(action_space.low, device=logits.device, cast=logits.dtype, in_place=False)
        self.high = convert_to_torch(action_space.high, device=logits.device, cast=logits.dtype, in_place=False)

        alpha, beta = torch.chunk(logits, 2, dim=-1)
        super().__init__(dist=Beta(concentration1=alpha, concentration0=beta), action_space=action_space)

    @override(TorchProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        un_squashed_values = self._un_squash(actions)
        log_prob = self.dist.log_prob(un_squashed_values).mean(-1)
        assert self.dist.concentration0.shape[:-1] == log_prob.shape
        assert self.dist.concentration1.shape[:-1] == log_prob.shape
        return log_prob

    @override(TorchProbabilityDistribution)
    def deterministic_sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return self._squash(self.dist.mean)

    @override(TorchProbabilityDistribution)
    def sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        normal_sample = self.dist.rsample()
        return self._squash(normal_sample)

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

    def _squash(self, raw_values: torch.Tensor) -> torch.Tensor:
        return raw_values * (self.high - self.low) + self.low

    def _un_squash(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.low) / (self.high - self.low)
