"""Contains a Categorical distribution."""
from typing import Sequence

import torch
from gym import spaces
from torch.distributions import Categorical

from maze.core.annotations import override
from maze.distributions.torch_dist import TorchProbabilityDistribution


class CategoricalProbabilityDistribution(TorchProbabilityDistribution[Categorical]):
    """Categorical Torch probability distribution.

    :param logits: the action selection logits.
    """

    @classmethod
    @override(TorchProbabilityDistribution)
    def required_logits_shape(cls, action_space: spaces.Discrete) -> Sequence[int]:
        """implementation of :class:`~maze.distributions.torch_dist.TorchProbabilityDistribution` interface
        """
        return [action_space.n]

    def __init__(self, logits: torch.Tensor, action_space: spaces.Discrete, temperature: float):
        self.logits = logits / temperature
        super().__init__(dist=Categorical(logits=self.logits), action_space=action_space)

    @override(TorchProbabilityDistribution)
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        log_prob = super().log_prob(actions)
        assert self.dist.logits.shape[:-1] == log_prob.shape
        return log_prob

    @override(TorchProbabilityDistribution)
    def deterministic_sample(self):
        """implementation of :class:`~maze.distributions.distribution.ProbabilityDistribution` interface
        """
        return self.logits.argmax(dim=-1)
