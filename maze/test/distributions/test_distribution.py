"""Contains distribution tests."""
import numpy as np
import pytest
import torch
from gym import spaces

from maze.distributions.bernoulli import BernoulliProbabilityDistribution
from maze.distributions.distribution import ProbabilityDistribution


def test_neg_logprob():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    sample = dist.sample()
    assert np.allclose(dist.neg_log_prob(actions=sample).numpy(), -dist.log_prob(actions=sample).numpy())


def test_not_implemented():
    """Test if all NotImplementedError are raise correctly"""
    pd = ProbabilityDistribution()

    with pytest.raises(NotImplementedError):
        pd.log_prob(None)

    with pytest.raises(NotImplementedError):
        pd.entropy()

    with pytest.raises(NotImplementedError):
        pd.kl(ProbabilityDistribution())

    with pytest.raises(NotImplementedError):
        pd.sample()

    with pytest.raises(NotImplementedError):
        pd.deterministic_sample()




