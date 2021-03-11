"""Contains distribution tests."""
import numpy as np
import torch
from gym import spaces

from maze.distributions.multi_categorical import MultiCategoricalProbabilityDistribution


def test_multi_categorical_sample():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    for multi_sample in [dist.sample(), dist.deterministic_sample()]:
        for sample in multi_sample:
            assert sample.numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    for multi_sample in [dist.sample(), dist.deterministic_sample()]:
        assert multi_sample.numpy().ndim == 2
        assert multi_sample.numpy().shape == (100, 2)

    logits = torch.from_numpy(np.random.randn(100, 12, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    for multi_sample in [dist.sample(), dist.deterministic_sample()]:
        assert multi_sample.numpy().ndim == 3
        assert multi_sample.numpy().shape == (100, 12, 2)


def test_multi_categorical_entropy():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.entropy().numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.entropy().numpy().ndim == 1
    assert dist.entropy().numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.entropy().numpy().ndim == 2
    assert dist.entropy().numpy().shape == (100, 8)


def test_multi_categorical_logprob():
    """ distribution test """
    logits = torch.from_numpy(np.random.randn(8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.neg_log_prob(actions=dist.sample()).numpy().ndim == 0

    logits = torch.from_numpy(np.random.randn(100, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.neg_log_prob(actions=dist.sample()).numpy().ndim == 1
    assert dist.neg_log_prob(actions=dist.sample()).numpy().shape == (100,)

    logits = torch.from_numpy(np.random.randn(100, 8, 8))
    dist = MultiCategoricalProbabilityDistribution(logits=logits, action_space=spaces.MultiDiscrete((3, 5)),
                                                   temperature=1.0)
    assert dist.neg_log_prob(actions=dist.sample()).numpy().ndim == 2
    assert dist.neg_log_prob(actions=dist.sample()).numpy().shape == (100, 8)


def test_multi_categorical_kl():
    """ distribution test """
    logits_0 = torch.from_numpy(np.random.randn(100, 8, 8))
    logits_1 = torch.from_numpy(np.random.randn(100, 8, 8))

    dist_0 = MultiCategoricalProbabilityDistribution(logits=logits_0, action_space=spaces.MultiDiscrete((3, 5)),
                                                     temperature=1.0)
    dist_1 = MultiCategoricalProbabilityDistribution(logits=logits_1, action_space=spaces.MultiDiscrete((3, 5)),
                                                     temperature=1.0)

    assert dist_0.kl(dist_1).numpy().ndim == 2
    assert dist_0.kl(dist_1).numpy().shape == (100, 8)


def test_multi_categorical_required_logits_shape():
    """ distribution test """
    shape = MultiCategoricalProbabilityDistribution.required_logits_shape(action_space=spaces.MultiDiscrete((3, 5)))
    assert shape == [8]
