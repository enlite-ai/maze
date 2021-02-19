"""This files contains the maze compatible custom RLlib ActionDistribution"""
from typing import Dict, Union

import gym
import numpy as np
import torch
from gym import spaces
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType

from maze.distributions.distribution_mapper import DistributionMapper
from maze.rllib.maze_rllib_models.maze_rllib_base_model import MazeRLlibBaseModel


class MazeRLlibActionDistribution(TorchDistributionWrapper):
    """Initializes a Custom Maze Action Distribution.

    An action distribution capable of dealing with envs that have dict action spaces, and individual,
    action-head-specific output distributions. To achieve this, the
    :class:`~maze.distributions.distribution_mapper.DistributionMapper` is utilized, which ensures full flexibility for
    specifying different distributions to the same gym action space type. (e.g. One gym.spaces.Box space could be
    modeled with a Beta another one with a DiagonalGaussian distribution.) It allows to add and register arbitrary
    custom distributions.

    :param inputs: A single tensor of shape [BATCH, size].
        Here BATCH represents the batch dimension and size the dimension of all concatenated output logits.
    :param model: The Union[ModelV2, MazeRLlibModel] model used to predict logits for this distribution.
    """

    @override(TorchDistributionWrapper)
    def __init__(self, inputs: torch.Tensor, model: MazeRLlibBaseModel, temperature: float = 1.0):
        assert isinstance(model, TorchModelV2), f'Expected TorchModelV2 but got {type(model)} with bases ' \
                                                f'{model.__class__.__bases__}'

        super().__init__(inputs, model)

        assert isinstance(inputs, torch.Tensor)
        assert len(inputs.shape) > 1, 'Batch dimension has to be given'

        # Retrieve maze distribution mapper from maze-model-composer
        distribution_mapper = model.model_composer.distribution_mapper
        for key, space in distribution_mapper.action_space.spaces.items():
            assert not isinstance(space, spaces.MultiBinary), 'Multi-binary action space not yet supported'
            assert not isinstance(space, spaces.MultiDiscrete), 'Multi-discrete action space not yet supported'

        # Retrieve and save the sorted action keys
        self.action_heads = list(sorted(distribution_mapper.action_space.spaces.keys()))

        # Compute the split sizes in order to split the input tensor according to the actual actions (and their
        #   individual sizes)
        self.split_sizes = [np.prod(distribution_mapper.required_logits_shape(action_head))
                            for action_head in self.action_heads]
        # Compute the logits dict by splitting the input
        logits_dict = {k: v for k, v in zip(self.action_heads, inputs.split(self.split_sizes, dim=-1))}

        # Create the maze distribution
        self.maze_dist = distribution_mapper.logits_dict_to_distribution(logits_dict, temperature)

        self._batch_size = inputs.shape[0]
        self._batch_larger_1 = self._batch_size > 1
        self._split_sizes_action = [int(np.prod(sub_action.shape[int(self._batch_larger_1):]))
                                    for sub_action in self.sample().values()]

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.spaces.Dict, model_config: Dict) -> int:
        """Returns the required logits shape (network output shape) for a given action head.

        :param action_space: The action space of the env.
        :param model_config: The rllib model config.
        :return: The number of the flattened output.
        """
        # Retrieve the distribution_mapper_config from the model config
        method_distribution_mapper_config = \
            model_config['custom_model_config']['maze_model_composer_config']['distribution_mapper_config']
        # Build the distribution mapper
        method_distribution_mapper = DistributionMapper(action_space,
                                                        distribution_mapper_config=method_distribution_mapper_config)
        # Compute the flattened number of logits
        num_outputs = sum([np.prod(method_distribution_mapper.required_logits_shape(action_head)) for action_head in
                           method_distribution_mapper.action_space.spaces])
        return num_outputs

    @override(TorchDistributionWrapper)
    def logp(self, actions: Union[Dict[str, torch.Tensor], torch.Tensor]) -> TensorType:
        """Returns the the log likelihood of the provided actions.

        actions: The actions.
        :return: Log likelihood tensor.
        """
        if not isinstance(actions, dict):
            actions_dict = {k: v.squeeze() for k, v in zip(self.action_heads,
                                                           actions.split(self._split_sizes_action, dim=-1))}
        else:
            actions_dict = actions
        logp_dict = self.maze_dist.log_prob(actions_dict)

        for logp_action in logp_dict.values():
            assert logp_action.shape == torch.Size([self._batch_size])

        action_concat = torch.cat([v.unsqueeze(-1) for v in logp_dict.values()], dim=-1)
        logp = torch.sum(action_concat, dim=-1)
        assert logp.shape == torch.Size([self._batch_size])

        return logp

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        """Calculate the entropy of the probability distribution.

        :return: Entropy tensor.
        """
        return self.maze_dist.entropy()

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        """Calculates the Kullback-Leibler between self and the other probability distribution.

        :param other: The distribution to compare with.
        :return: Kl tensor.
        """
        return self.maze_dist.kl(other.maze_dist)

    @override(TorchDistributionWrapper)
    def sample(self) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Draw a sample from the probability distribution.

        :return: Stochastic sample tensor.
        """
        self.last_sample = self.maze_dist.sample()
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        """Returns the the log likelihood of the last sampled action.

        :return: Log likelihood tensor.
        """
        assert self.last_sample is not None
        return self.logp(self.last_sample)

    @override(ActionDistribution)
    def deterministic_sample(self) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Draw a deterministic sample from the probability distribution.

        :return: Deterministic sample tensor.
        """
        self.last_sample = self.maze_dist.deterministic_sample()
        return self.last_sample
