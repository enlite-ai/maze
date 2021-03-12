"""Contains a squashed diagonal Gaussian distributions"""
from typing import Sequence

import numpy as np
import gym
import torch
from torch.distributions import Normal, kl_divergence

from maze.core.annotations import override
from maze.distributions.torch_dist import TorchProbabilityDistribution
from maze.distributions.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, EPSILON, tensor_clamp, atanh
from maze.perception.perception_utils import convert_to_torch


class SquashedGaussianProbabilityDistribution(TorchProbabilityDistribution[Normal]):
    """Tanh-squashed diagonal Gaussian (Normal) Torch probability distribution.

    :param logits: the logits for both mean and standard deviation.
    :param action_space: the underlying gym.spaces action space.
    """

    @classmethod
    @override(TorchProbabilityDistribution)
    def required_logits_shape(cls, action_space: gym.spaces.Space) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        required_shape = [2 * v for v in action_space.shape]
        return required_shape

    def __init__(self, logits: torch.Tensor, action_space: gym.spaces.Box, temperature: float):

        # make sure space bounds are set properly
        assert np.all(action_space.low > -np.inf) and np.all(action_space.low < np.inf)

        # create low and high tensors for vectorized clamping
        self.low = convert_to_torch(action_space.low, device=logits.device, cast=logits.dtype, in_place=False)
        self.high = convert_to_torch(action_space.high, device=logits.device, cast=logits.dtype, in_place=False)

        # split distribution parameters
        mean, log_std = torch.chunk(logits, 2, dim=-1)

        # apply the temperature to the standard deviation
        log_std = log_std / temperature

        # transform to reasonable range
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)

        super().__init__(dist=Normal(mean, std), action_space=action_space)

    @override(TorchProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        un_squashed_values = self._un_squash(actions)
        log_prob_gaussian = self.dist.log_prob(un_squashed_values)
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = log_prob_gaussian.mean(-1)
        un_squashed_values_tanh = torch.tanh(un_squashed_values)
        log_prob = log_prob_gaussian - torch.log(1 - un_squashed_values_tanh ** 2 + EPSILON).mean(-1)
        assert self.dist.mean.shape[:-1] == log_prob.shape
        return log_prob

    @override(TorchProbabilityDistribution)
    def deterministic_sample(self):
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return self._squash(self.dist.mean)

    @override(TorchProbabilityDistribution)
    def sample(self):
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
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * (self.high - self.low) + self.low
        return tensor_clamp(squashed, self.low, self.high)

    def _un_squash(self, values: torch.Tensor) -> torch.Tensor:
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        save_normed_values = torch.clamp(normed_values, -1.0 + EPSILON, 1.0 - EPSILON)
        un_squashed = atanh(save_normed_values)
        return un_squashed
