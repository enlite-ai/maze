"""Contains a dictionary (collection) distributions enclosing multiple actual sub-distributions."""
from typing import Dict

import torch

from maze.core.annotations import override
from maze.distributions.distribution import ProbabilityDistribution


class DictProbabilityDistribution(ProbabilityDistribution):
    """Dictionary probability distribution.

    The respective functions either return
        - the per key distribution properties or
        - aggregate the properties across the sub-distributions using a reduce_fun such as mean or sum.

    :param distribution_dict: dictionary holding sub-probability distributions.
    """

    def __init__(self, distribution_dict: Dict[str, ProbabilityDistribution]):
        super().__init__()
        self.distribution_dict = distribution_dict

    @override(ProbabilityDistribution)
    def neg_log_prob(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        neg_log_prob = self.log_prob(actions)
        for k, lp in neg_log_prob.items():
            neg_log_prob[k] *= -1
        return neg_log_prob

    @override(ProbabilityDistribution)
    def log_prob(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        log_prob = dict()
        for k, dist in self.distribution_dict.items():
            log_prob[k] = dist.log_prob(actions[k])
        return log_prob

    @override(ProbabilityDistribution)
    def entropy(self, reduce_fun: callable = torch.mean) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        # collect and stack entropy of individual distributions
        entropy_list = []
        for k, dist in self.distribution_dict.items():
            entropy_list.append(dist.entropy())
        entropy_list = torch.stack(entropy_list)
        assert entropy_list.shape[0] == len(self.distribution_dict)

        return reduce_fun(entropy_list, dim=0)

    @override(ProbabilityDistribution)
    def kl(self, other: 'DictProbabilityDistribution', reduce_fun: callable = torch.mean) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        # collect and stack kls of individual distributions
        kl_list = []
        for k in self.distribution_dict.keys():
            kl_list.append(self.distribution_dict[k].kl(other.distribution_dict[k]))
        entropy_list = torch.stack(kl_list)
        assert entropy_list.shape[0] == len(self.distribution_dict)

        return reduce_fun(entropy_list, dim=0)

    @override(ProbabilityDistribution)
    def sample(self) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        sample = dict()
        for k, dist in self.distribution_dict.items():
            sample[k] = dist.sample()
        return sample

    @override(ProbabilityDistribution)
    def deterministic_sample(self) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        sample = dict()
        for k, dist in self.distribution_dict.items():
            sample[k] = dist.deterministic_sample()
        return sample
