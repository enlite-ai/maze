"""Contains distribution tests."""
import numpy as np
import torch
from gym import spaces

from maze.distributions.categorical import CategoricalProbabilityDistribution


def test_categorical_sample():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.sample().numpy().ndim == 0
    assert dist.deterministic_sample().numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.sample().numpy().ndim == 1
    assert dist.deterministic_sample().numpy().ndim == 1
    assert dist.sample().numpy().shape == (100,)
    assert dist.deterministic_sample().numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.sample().numpy().ndim == 2
    assert dist.deterministic_sample().numpy().ndim == 2
    assert dist.sample().numpy().shape == (100, 8)
    assert dist.deterministic_sample().numpy().shape == (100, 8)


def test_categorical_entropy():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.entropy().numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.entropy().numpy().ndim == 1
    assert dist.entropy().numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.entropy().numpy().ndim == 2
    assert dist.entropy().numpy().shape == (100, 8)


def test_categorical_logprob():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 1
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 5))
    dist = CategoricalProbabilityDistribution(logits=logits, action_space=spaces.Discrete(5), temperature=1.0)
    assert dist.log_prob(actions=dist.sample()).numpy().ndim == 2
    assert dist.log_prob(actions=dist.sample()).numpy().shape == (100, 8)


def test_bernoulli_kl():
    """ distribution test """
    logits_0 = torch.from_numpy(np.random.randn(100, 8, 5))
    logits_1 = torch.from_numpy(np.random.randn(100, 8, 5))

    dist_0 = CategoricalProbabilityDistribution(logits=logits_0, action_space=spaces.Discrete(5), temperature=1.0)
    dist_1 = CategoricalProbabilityDistribution(logits=logits_1, action_space=spaces.Discrete(5), temperature=1.0)

    assert dist_0.kl(dist_1).numpy().ndim == 2
    assert dist_0.kl(dist_1).numpy().shape == (100, 8)


def test_categorical_required_logits_shape():
    """ distribution test """
    shape = CategoricalProbabilityDistribution.required_logits_shape(action_space=spaces.Discrete(5))
    assert shape == [5]
