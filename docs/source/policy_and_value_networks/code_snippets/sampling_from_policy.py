"""Minimum working example showing how to sample actions from a policy network."""
from typing import Dict, Sequence

import torch
from gymnasium import spaces
from torch import nn

from maze.distributions.distribution_mapper import DistributionMapper

OBSERVATION_NAME = 'my_observation'
ACTION_NAME = 'my_action'


class PolicyNet(nn.Module):
    """Simple feed forward policy network."""

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Sequence[int]]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_shapes[OBSERVATION_NAME][0], out_features=16), nn.Tanh(),
            nn.Linear(in_features=16, out_features=action_logits_shapes[ACTION_NAME][0]))

    def forward(self, in_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ forward pass. """
        return {ACTION_NAME: self.net(in_dict[OBSERVATION_NAME])}


# init default distribution mapper
distribution_mapper = DistributionMapper(
    action_space=spaces.Dict(spaces={ACTION_NAME: spaces.Discrete(2)}),
    distribution_mapper_config={})

# request required action logits shape and init a policy net
logits_shape = distribution_mapper.required_logits_shape(ACTION_NAME)
policy_net = PolicyNet(obs_shapes={OBSERVATION_NAME: (4,)},
                       action_logits_shapes={ACTION_NAME: logits_shape})

# compute action logits (here from random input)
logits_dict = policy_net({OBSERVATION_NAME: torch.randn(4)})

# init action sampling distribution from model output
dist = distribution_mapper.logits_dict_to_distribution(logits_dict, temperature=1.0)

# sample action (e.g., {my_action: 1})
action = dist.sample()
