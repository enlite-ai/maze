"""Dummy implementation of a value net for the dummy env"""

from typing import Dict, Sequence

import torch
from torch import nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from maze.test.shared_test_utils.dummy_models.base_model import DummyBaseNet


class DummyValueNet(DummyBaseNet):
    """Policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], non_lin: type(nn.Module)):
        super().__init__(obs_shapes, non_lin)

        self.perception_dict['value_head_net'] = DenseBlock(
            in_keys='hidden_out', in_shapes=self.perception_dict['hidden_out'].out_shapes(),
            out_keys='value_head_net', hidden_units=[5, 2], non_lin=non_lin)

        self.perception_dict['value'] = LinearOutputBlock(
            in_keys='value_head_net', in_shapes=self.perception_dict['value_head_net'].out_shapes(),
            out_keys='value', output_units=1)

        # Set up inference block
        self.perception_net = InferenceBlock(
            in_keys=list(self.obs_shapes.keys()), out_keys='value',
            in_shapes=[self.obs_shapes[key] for key in self.obs_shapes.keys()],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['value'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param xx: Input dict.
        :return: The computed output of the network.
        """
        return self.perception_net(xx)
