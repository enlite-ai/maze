"""Contains a multi-categorical distributions enclosing multiple categorical distributions."""
from typing import Sequence

import torch
from gym import spaces

from maze.core.annotations import override
from maze.distributions.categorical import CategoricalProbabilityDistribution
from maze.distributions.distribution import ProbabilityDistribution


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    """Multi-categorical probability distribution.

    The respective functions either return aggregated properties across the sub-distributions
    using a reduce_fun such as mean or sum.

    :param logits: The concatenated action selection logits for all sub spaces.
    """

    @classmethod
    def required_logits_shape(cls, action_space: spaces.MultiDiscrete) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return [sum(action_space.nvec)]

    def __init__(self, logits: torch.Tensor, action_space: spaces.MultiDiscrete, temperature: float):

        # instantiate categorical sub-distributions
        self.sub_distributions = []
        i0 = 0
        for i, n in enumerate(action_space.nvec):
            sub_distribution = CategoricalProbabilityDistribution(logits=logits[..., i0:i0 + n],
                                                                  action_space=spaces.Discrete(action_space.nvec[i]),
                                                                  temperature=temperature)
            self.sub_distributions.append(sub_distribution)

            # shift logits starting index
            i0 += n

    @override(ProbabilityDistribution)
    def neg_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return -self.log_prob(actions)

    @override(ProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        log_prob = []
        for k, dist in enumerate(self.sub_distributions):
            log_prob.append(dist.log_prob(actions[..., k]))
        return torch.stack(log_prob, dim=0).mean(dim=0)

    @override(ProbabilityDistribution)
    def entropy(self, reduce_fun: callable = torch.mean) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        # collect and stack entropy of individual distributions
        entropy_list = [d.entropy() for d in self.sub_distributions]
        entropy_list = torch.stack(entropy_list)
        assert entropy_list.shape[0] == len(self.sub_distributions)

        return reduce_fun(entropy_list, dim=0)

    @override(ProbabilityDistribution)
    def kl(self, other: 'MultiCategoricalProbabilityDistribution', reduce_fun: callable = torch.mean) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        # collect and stack kls of individual distributions
        kl_list = []
        for i, dist in enumerate(self.sub_distributions):
            kl_list.append(self.sub_distributions[i].kl(other.sub_distributions[i]))
        kl_list = torch.stack(kl_list)
        assert kl_list.shape[0] == len(self.sub_distributions)

        return reduce_fun(kl_list, dim=0)

    @override(ProbabilityDistribution)
    def sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return torch.stack([d.sample() for d in self.sub_distributions], dim=-1)

    @override(ProbabilityDistribution)
    def deterministic_sample(self) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return torch.stack([d.deterministic_sample() for d in self.sub_distributions], dim=-1)
