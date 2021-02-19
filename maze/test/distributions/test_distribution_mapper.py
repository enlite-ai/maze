"""Contains distribution tests."""
import numpy as np
import torch
from gym import spaces

from maze.distributions.bernoulli import BernoulliProbabilityDistribution
from maze.distributions.beta import BetaProbabilityDistribution
from maze.distributions.categorical import CategoricalProbabilityDistribution
from maze.distributions.dict import DictProbabilityDistribution
from maze.distributions.distribution_mapper import DistributionMapper
from maze.distributions.squashed_gaussian import SquashedGaussianProbabilityDistribution


def test_distribution_mapper():
    """ distribution test """

    # action space
    act_space = spaces.Dict(spaces={"selection": spaces.Discrete(10),
                                    "order": spaces.MultiBinary(15),
                                    "scale_input": spaces.Box(shape=(5,), low=0, high=100, dtype=np.float64),
                                    "order_by_weight": spaces.Box(shape=(5,), low=0, high=100, dtype=np.float64)})

    # default config
    config = [
        {"action_space": spaces.Box,
         "distribution": "maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution"},
        {"action_head": "order_by_weight",
         "distribution": "maze.distributions.beta.BetaProbabilityDistribution"}
    ]

    # initialize distribution mapper
    distribution_mapper = DistributionMapper(action_space=act_space,
                                             distribution_mapper_config=config)
    repr(distribution_mapper)

    # assign action heads to registered distributions
    logits_dict = dict()
    for action_head in act_space.spaces.keys():
        logits_shape = distribution_mapper.required_logits_shape(action_head)

        logits_tensor = torch.from_numpy(np.random.randn(*logits_shape))
        torch_dist = distribution_mapper.action_head_distribution(action_head=action_head, logits=logits_tensor,
                                                                  temperature=1.0)
        logits_dict[action_head] = logits_tensor

        # check if distributions are correctly assigned
        if action_head == "selection":
            assert isinstance(torch_dist, CategoricalProbabilityDistribution)
        elif action_head == "order":
            assert isinstance(torch_dist, BernoulliProbabilityDistribution)
        elif action_head == "scale_input":
            assert isinstance(torch_dist, SquashedGaussianProbabilityDistribution)
        elif action_head == "order_by_weight":
            assert isinstance(torch_dist, BetaProbabilityDistribution)

    # test dictionary distribution mapping
    dict_dist = distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict, temperature=1.0)
    assert isinstance(dict_dist, DictProbabilityDistribution)
