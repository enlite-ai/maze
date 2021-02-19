"""Contains distribution tests."""
import numpy as np
import torch
from gym import spaces

from maze.distributions.bernoulli import BernoulliProbabilityDistribution


def test_bernoulli_sample():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.sample().numpy().ndim == 1

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.sample().numpy().ndim == 2
    assert dist.sample().numpy().shape == (100, 5)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.sample().numpy().ndim == 3
    assert dist.sample().numpy().shape == (100, 8, 5)


def test_bernoulli_deterministic():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5, 7, 11))

    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert np.array_equal(dist.deterministic_sample().numpy(), (logits > 0).numpy())


def test_bernoulli_entropy():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.entropy().numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.entropy().numpy().ndim == 1
    assert dist.entropy().numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.entropy().numpy().ndim == 2
    assert dist.entropy().numpy().shape == (100, 8)


def test_bernoulli_logprob():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 1

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 2
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100, 5)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = BernoulliProbabilityDistribution(logits=logits, action_space=spaces.MultiBinary(5))
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 3
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100, 8, 5)


def test_bernoulli_kl():
    """ distribution test """
    logits_0 = torch.from_numpy(np.random.randn(100, 8, 5))
    logits_1 = torch.from_numpy(np.random.randn(100, 8, 5))

    dist_0 = BernoulliProbabilityDistribution(logits=logits_0, action_space=spaces.MultiBinary(5))
    dist_1 = BernoulliProbabilityDistribution(logits=logits_1, action_space=spaces.MultiBinary(5))

    assert dist_0.kl(dist_1).numpy().ndim == 2
    assert dist_0.kl(dist_1).numpy().shape == (100, 8)


def test_bernoulli_required_logits_shape():
    """ distribution test """
    shape = BernoulliProbabilityDistribution.required_logits_shape(action_space=spaces.MultiBinary(5))
    assert shape == [5]
