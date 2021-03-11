"""Shows how to use the custom model composer to build a custom policy network."""
from collections import OrderedDict
from typing import Dict, Union, Sequence, List

import numpy as np
import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class CustomCartpolePolicyNet(nn.Module):
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: A list of units per hidden layer.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], hidden_units: List[int]):
        super().__init__()

        # Maze relies on dictionaries to represent the inference graph
        self.perception_dict = OrderedDict()

        # build latent embedding block
        self.perception_dict['latent'] = DenseBlock(
            in_keys='observation', out_keys='latent', in_shapes=obs_shapes['observation'],
            hidden_units=hidden_units,non_lin=non_lin)

        # build action head
        self.perception_dict['action'] = LinearOutputBlock(
            in_keys='latent', out_keys='action', in_shapes=self.perception_dict['latent'].out_shapes(),
            output_units=int(np.prod(action_logits_shapes["action"])))

        # build inference block
        self.perception_net = InferenceBlock(
            in_keys='observation', out_keys='action', in_shapes=obs_shapes['observation'],
            perception_blocks=self.perception_dict)

        # apply weight init
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['action'].apply(make_module_init_normc(0.01))

    def forward(self, in_tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param in_tensor_dict: Input tensor dict.
        :return: The computed output of the network.
        """
        return self.perception_net(in_tensor_dict)
