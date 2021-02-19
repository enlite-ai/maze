"""Contains distribution tests."""
import numpy as np
import torch
from gym import spaces

from maze.distributions.bernoulli import BernoulliProbabilityDistribution
from maze.distributions.categorical import CategoricalProbabilityDistribution
from maze.distributions.dict import DictProbabilityDistribution


def get_dict_distribution():
    """ distribution test """
    logits_0 = torch.from_numpy(np.random.randn(100, 3))
    logits_1 = torch.from_numpy(np.random.randn(100, 5))

    distribution_dict = {"action_0": CategoricalProbabilityDistribution(logits=logits_0,
                                                                        action_space=spaces.Discrete(9),
                                                                        temperature=1.0),
                         "action_1": BernoulliProbabilityDistribution(logits=logits_1,
                                                                      action_space=spaces.MultiBinary(5),
                                                                      temperature=1.0)}

    return DictProbabilityDistribution(distribution_dict=distribution_dict)


def test_dict_sample():
    """ distribution test """
    dist = get_dict_distribution()
    for sample in [dist.sample(), dist.deterministic_sample()]:

        assert sample["action_0"].numpy().ndim == 1
        assert sample["action_0"].numpy().shape == (100,)

        assert sample["action_1"].numpy().ndim == 2
        assert sample["action_1"].numpy().shape == (100, 5)


def test_dict_entropy():
    """ distribution test """
    dist = get_dict_distribution()
    assert dist.entropy().numpy().ndim == 1
    assert dist.entropy().numpy().shape == (100,)


def test_dict_neg_logprob():
    """ distribution test """
    dist = get_dict_distribution()
    actions = dist.sample()

    log_probs = dist.neg_log_prob(actions=actions)

    assert log_probs["action_0"].numpy().ndim == 1
    assert log_probs["action_0"].numpy().shape == (100,)

    assert log_probs["action_1"].numpy().ndim == 2
    assert log_probs["action_1"].numpy().shape == (100, 5)


def test_dict_kl():
    """ distribution test """
    dist_0 = get_dict_distribution()
    dist_1 = get_dict_distribution()
    assert dist_0.kl(dist_1).numpy().ndim == 1
    assert dist_0.kl(dist_1).numpy().shape == (100,)
