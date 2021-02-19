"""Test the functionality of the dummy model on the dummy env"""

import torch
import torch.nn as nn
from gym import spaces

from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import observation_spaces_to_in_shapes
from maze.test.shared_test_utils.dummy_models.actor_model import DummyPolicyNet
from maze.test.shared_test_utils.dummy_models.critic_model import DummyValueNet
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_dummy_model_with_dummy_network():
    """
    Unit test for the DummyStructuredEnvironment
    """
    maze_env = build_dummy_maze_env()

    # init the distribution_mapper with the flat action space
    distribution_mapper_config = [
        {"action_space": spaces.Box,
         "distribution": "maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution"}
    ]
    distribution_mapper = DistributionMapper(action_space=maze_env.action_space,
                                             distribution_mapper_config=distribution_mapper_config)

    obs_shapes = observation_spaces_to_in_shapes(maze_env.observation_spaces_dict)

    dummy_actor = DummyPolicyNet(obs_shapes=obs_shapes[0],
                                 action_logits_shapes={key: distribution_mapper.required_logits_shape(key)
                                                       for key in maze_env.action_space.spaces.keys()},
                                 non_lin=nn.Tanh)

    dummy_critic = DummyValueNet(obs_shapes=obs_shapes[0], non_lin=nn.Tanh)

    obs_np = maze_env.reset()
    obs = {k: torch.from_numpy(v) for k, v in obs_np.items()}

    for i in range(100):
        logits_dict = dummy_actor(obs)
        prob_dist = distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict, temperature=1.0)
        sampled_actions = prob_dist.sample()

        obs_np, _, _, _ = maze_env.step(sampled_actions)
        obs = {k: torch.from_numpy(v) for k, v in obs_np.items()}

        _ = dummy_critic(obs)
    maze_env.close()
