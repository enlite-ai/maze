"""Contains distribution tests."""
import numpy as np
import torch
from gym.spaces import Box

from maze.distributions.beta import BetaProbabilityDistribution


def test_beta_sample():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.sample().numpy().ndim == 1
    assert dist.deterministic_sample().numpy().ndim == 1

    logits = torch.from_numpy(np.random.randn(100, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.sample().numpy().ndim == 2
    assert dist.deterministic_sample().numpy().ndim == 2
    assert dist.sample().numpy().shape == (100, 3)
    assert dist.deterministic_sample().numpy().shape == (100, 3)

    logits = torch.from_numpy(np.random.randn(100, 8, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.sample().numpy().ndim == 3
    assert dist.deterministic_sample().numpy().ndim == 3
    assert dist.sample().numpy().shape == (100, 8, 3)
    assert dist.deterministic_sample().numpy().shape == (100, 8, 3)


def test_beta_entropy():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.entropy().numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(100, 3), dtype=np.float64),
                                       temperature=1.0)
    assert dist.entropy().numpy().ndim == 1
    assert dist.entropy().numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.entropy().numpy().ndim == 2
    assert dist.entropy().numpy().shape == (100, 8)


def test_beta_logprob():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 1
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 6))
    dist = BetaProbabilityDistribution(logits=logits,
                                       action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                       temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 2
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100, 8)


def test_beta_kl():
    """ distribution test """
    logits_0 = torch.from_numpy(np.random.randn(100, 8, 6))
    logits_1 = torch.from_numpy(np.random.randn(100, 8, 6))

    dist_0 = BetaProbabilityDistribution(logits=logits_0,
                                         action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                         temperature=1.0)
    dist_1 = BetaProbabilityDistribution(logits=logits_1,
                                         action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64),
                                         temperature=1.0)

    assert dist_0.kl(dist_1).numpy().ndim == 2
    assert dist_0.kl(dist_1).numpy().shape == (100, 8)


def test_beta_required_logits_shape():
    """ distribution test """
    shape = BetaProbabilityDistribution.required_logits_shape(
        action_space=Box(low=0.0, high=10.0, shape=(3,), dtype=np.float64))
    assert shape == [6]
