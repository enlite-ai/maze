"""Shows how to use the custom model composer to build a complex custom embedding networks."""
from collections import OrderedDict
from typing import Dict, Union, Sequence, List

import torch.nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.joint_blocks.lstm_last_step import LSTMLastStepBlock
from maze.perception.blocks.joint_blocks.vgg_conv_dense import VGGConvolutionDenseBlock


class CustomComplexLatentNet:
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: A list of units per hidden layer.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], hidden_units: List[int]):
        self.obs_shapes = obs_shapes

        # Maze relies on dictionaries to represent the inference graph
        self.perception_dict = OrderedDict()

        # build latent feature embedding block
        self.perception_dict['latent_inventory'] = DenseBlock(
            in_keys='observation_inventory', out_keys='latent_inventory', in_shapes=obs_shapes['observation_inventory'],
            hidden_units=[128], non_lin=non_lin)

        # build latent pixel embedding block
        self.perception_dict['latent_screen'] = VGGConvolutionDenseBlock(
            in_keys='observation_screen', out_keys='latent_screen', in_shapes=obs_shapes['observation_screen'],
            non_lin=non_lin, hidden_channels=[8, 16, 32], hidden_units=[32], use_batch_norm_conv=False)

        # Concatenate latent features
        self.perception_dict['latent_concat'] = ConcatenationBlock(
            in_keys=['latent_inventory', 'latent_screen'], out_keys='latent_concat',
            in_shapes=self.perception_dict['latent_inventory'].out_shapes() +
            self.perception_dict['latent_screen'].out_shapes(), concat_dim=-1)

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
