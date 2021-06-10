"""Contains maze-rllib-action-distribution tests."""
import random

import numpy as np
import pytest
import torch
from gym import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import convert_to_torch
from maze.rllib.maze_rllib_action_distribution import MazeRLlibActionDistribution


class FakeRLLibModel(TorchModelV2, torch.nn.Module):
    """Fake RLlibModel"""

    def __init__(self, distribution_mapper: DistributionMapper):
        torch.nn.Module.__init__(self)
        TorchModelV2.__init__(self, action_space=None, num_outputs=1, model_config={}, name='',
                              obs_space=None)

        class FakeModelComposer:
            """Fake Model composer"""
            def __init__(self2, distribution_mapper: DistributionMapper):
                self2.distribution_mapper = distribution_mapper

        self.model_composer = FakeModelComposer(distribution_mapper)


@pytest.mark.rllib
def test_maze_rllib_action_dist_batch_0():
    perform_test_maze_rllib_action_distribution(0)


@pytest.mark.rllib
def test_maze_rllib_action_dist_batch_20():
    perform_test_maze_rllib_action_distribution(20)


@pytest.mark.rllib
def perform_test_maze_rllib_action_distribution(batch_dim: int):
    """ distribution test """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # action space
    act_space = spaces.Dict(spaces=dict(sorted(
        {"selection": spaces.Discrete(10),
         "scale_input": spaces.Box(shape=(5,), low=0, high=100, dtype=np.float64),
         "order_by_weight": spaces.Box(shape=(5,), low=0, high=100, dtype=np.float64)}.items())))

    # default config
    config = [
        {"action_space": spaces.Box,
         "distribution": "maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution"},
        {"action_head": "order_by_weight", "distribution": "maze.distributions.beta.BetaProbabilityDistribution"}
    ]

    # initialize distribution mapper
    distribution_mapper = DistributionMapper(action_space=act_space,
                                             distribution_mapper_config=config)

    num_outputs = sum([np.prod(distribution_mapper.required_logits_shape(action_head)) for action_head in
                       distribution_mapper.action_space.spaces])
    model_config = {'custom_model_config': {'maze_model_composer_config': {'distribution_mapper_config': config}}}
    assert num_outputs == MazeRLlibActionDistribution.required_model_output_shape(act_space, model_config)

    # assign action heads to registered distributions
    logits_dict = dict()
    for action_head in act_space.spaces.keys():

        logits_shape = distribution_mapper.required_logits_shape(action_head)
        if batch_dim > 0:
            logits_shape = (batch_dim, *logits_shape)

        logits_tensor = torch.from_numpy(np.random.randn(*logits_shape))
        logits_dict[action_head] = logits_tensor

    flat_input = torch.cat([tt for tt in logits_dict.values()], dim=-1)
    if batch_dim == 0:
        flat_input = flat_input.unsqueeze(0)
    fake_model = FakeRLLibModel(distribution_mapper)
    rllib_dist = MazeRLlibActionDistribution(flat_input, fake_model, temperature=0.5)

    # test dictionary distribution mapping
    maze_dist = distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict, temperature=0.5)

    for action_head in act_space.spaces.keys():
        maze_distribution = maze_dist.distribution_dict[action_head]
        maze_rllib_distribution = rllib_dist.maze_dist.distribution_dict[action_head]
        if hasattr(maze_distribution, 'logits'):
            assert torch.allclose(maze_distribution.logits,
                                  maze_rllib_distribution.logits)
        if hasattr(maze_distribution, 'low'):
            assert torch.allclose(maze_distribution.low,
                                  maze_rllib_distribution.low)
            assert torch.allclose(maze_distribution.high,
                                  maze_rllib_distribution.high)

    test_action_maze = maze_dist.sample()
    test_action_rllib = rllib_dist.sample()

    for action_head in act_space.spaces.keys():
        assert test_action_maze[action_head].shape == test_action_rllib[action_head].shape[int(batch_dim == 0):]

    maze_action = maze_dist.deterministic_sample()
    rllib_action = rllib_dist.deterministic_sample()

    for action_head in act_space.spaces.keys():
        assert torch.all(maze_action[action_head] == rllib_action[action_head])

    maze_action = convert_to_torch(maze_action, device=None, cast=torch.float64, in_place=True)
    rllib_action = convert_to_torch(rllib_action, device=None, cast=torch.float64, in_place=True)

    # This un-sqeeze is preformed by rllib before passing an action to log p
    for action_head in act_space.spaces.keys():
        if len(rllib_action[action_head].shape) == 0:
            rllib_action[action_head] = rllib_action[action_head].unsqueeze(0)

    logp_maze_dict = maze_dist.log_prob(maze_action)
    action_concat = torch.cat([v.unsqueeze(-1) for v in logp_maze_dict.values()], dim=-1)
    logp_maze = torch.sum(action_concat, dim=-1)

    logp_rllib = rllib_dist.logp(rllib_action)
    if batch_dim == 0:
        logp_rllib = logp_rllib[0]

    assert torch.equal(logp_maze, logp_rllib)

    logp_rllib_2 = rllib_dist.sampled_action_logp()
    if batch_dim == 0:
        logp_rllib_2 = logp_rllib_2[0]

    assert torch.equal(logp_maze, logp_rllib_2)

    maze_entropy = maze_dist.entropy()
    rllib_entropy = rllib_dist.entropy()
    if batch_dim == 0:
        rllib_entropy = rllib_entropy[0]

    assert torch.equal(maze_entropy, rllib_entropy)

    logits_dict2 = dict()
    for action_head in act_space.spaces.keys():
        logits_shape = distribution_mapper.required_logits_shape(action_head)
        if batch_dim > 0:
            logits_shape = (batch_dim, *logits_shape)

        logits_tensor = torch.from_numpy(np.random.randn(*logits_shape))
        logits_dict2[action_head] = logits_tensor

    flat_input = torch.cat([tt for tt in logits_dict2.values()], dim=-1)
    if batch_dim == 0:
        flat_input = flat_input.unsqueeze(0)
    fake_model = FakeRLLibModel(distribution_mapper)
    rllib_dist_2 = MazeRLlibActionDistribution(flat_input, fake_model, temperature=0.5)

    # test dictionary distribution mapping
    maze_dist_2 = distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict2, temperature=0.5)

    maze_kl = maze_dist.kl(maze_dist_2)
    rllib_kl = rllib_dist.kl(rllib_dist_2)
    if batch_dim == 0:
        rllib_kl = rllib_kl[0]

    assert torch.equal(maze_kl, rllib_kl)
