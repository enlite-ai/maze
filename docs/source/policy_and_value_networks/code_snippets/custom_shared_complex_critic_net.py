"""Shows how to use the custom model composer to build a complex custom value networks with shared embedding."""
from collections import OrderedDict
from typing import Dict, Union, Sequence, List

import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.joint_blocks.lstm_last_step import LSTMLastStepBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class CustomSharedComplexCriticNet(nn.Module):
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: A list of units per hidden layer.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], hidden_units: List[int]):
        nn.Module.__init__(self)

        # Maze relies on dictionaries to represent the inference graph
        self.perception_dict = OrderedDict()

        # build latent feature embedding block
        self.perception_dict['latent_inventory'] = DenseBlock(
            in_keys='observation_inventory', out_keys='latent_inventory', in_shapes=obs_shapes['observation_inventory'],
            hidden_units=[128], non_lin=non_lin)

        # Concatenate latent features
        self.perception_dict['latent_concat'] = ConcatenationBlock(
            in_keys=['latent_inventory', 'latent_screen'], out_keys='latent_concat',
            in_shapes=self.perception_dict['latent_inventory'].out_shapes() +
                      [obs_shapes['latent_screen']], concat_dim=-1)

        # Add latent dense block
        self.perception_dict['latent_dense'] = DenseBlock(
            in_keys='latent_concat', out_keys='latent_dense', hidden_units=hidden_units, non_lin=non_lin,
            in_shapes=self.perception_dict['latent_concat'].out_shapes()
        )

        # Add recurrent block
        self.perception_dict['latent'] = LSTMLastStepBlock(
            in_keys='latent_dense', out_keys='latent', in_shapes=self.perception_dict['latent_dense'].out_shapes(),
            hidden_size=32, num_layers=1, bidirectional=False, non_lin=non_lin
        )

        # build action heads
        self.perception_dict['value'] = LinearOutputBlock(
            in_keys='latent', out_keys='value', in_shapes=self.perception_dict['latent'].out_shapes(),
            output_units=1)

        # build inference block
        in_keys = list(obs_shapes.keys())
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys='value', in_shapes=[obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # apply weight init
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['value'].apply(make_module_init_normc(0.01))

    def forward(self, in_tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param in_tensor_dict: Input tensor dict.
        :return: The computed output of the network.
        """
        return self.perception_net(in_tensor_dict)
