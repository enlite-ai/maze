"""Cutting 2d critic network"""
from collections import OrderedDict
from typing import Dict, Union, Sequence

import torch
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from torch import nn as nn


class SelectionValueNet(nn.Module):
    """The Value net (critic) computing the predicted reward from the observations.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], non_lin: Union[str, type(nn.Module)]):
        nn.Module.__init__(self)
        self.obs_shapes = obs_shapes

        hidden_units = 32

        self.perception_dict = OrderedDict()

        self.perception_dict['inventory_feat'] = DenseBlock(
            in_keys='inventory-flatten', out_keys='inventory_feat', in_shapes=self.obs_shapes['inventory-flatten'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['order_feat'] = DenseBlock(
            in_keys='ordered_piece', out_keys='order_feat', in_shapes=self.obs_shapes['ordered_piece'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['latent'] = ConcatenationBlock(
            in_keys=['inventory_feat', 'order_feat'], out_keys='latent',
            in_shapes=[[hidden_units], [hidden_units]], concat_dim=-1)

        self.perception_dict['value'] = LinearOutputBlock(
            in_keys='latent', out_keys='value', in_shapes=self.perception_dict['latent'].out_shapes(), output_units=1)

        in_keys = ['inventory-flatten', 'ordered_piece']
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys='value',
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['value'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network

        :param xx: Input dict
        :return: The computed output of the network
        """
        return self.perception_net(xx)


class CuttingValueNet(nn.Module):
    """The Value net (critic) computing the predicted reward from the observations.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], non_lin: Union[str, type(nn.Module)]):
        nn.Module.__init__(self)
        self.obs_shapes = obs_shapes

        hidden_units = 32

        self.perception_dict = OrderedDict()

        self.perception_dict['order_feat'] = DenseBlock(
            in_keys='ordered_piece', out_keys='order_feat', in_shapes=self.obs_shapes['ordered_piece'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['selected_feat'] = DenseBlock(
            in_keys='selected_piece', out_keys='selected_feat', in_shapes=self.obs_shapes['selected_piece'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['latent'] = ConcatenationBlock(
            in_keys=['order_feat', 'selected_feat'], out_keys='latent',
            in_shapes=[[hidden_units], [hidden_units], [hidden_units]], concat_dim=-1)

        self.perception_dict['value'] = LinearOutputBlock(
            in_keys='latent', out_keys='value', in_shapes=self.perception_dict['latent'].out_shapes(), output_units=1)

        in_keys = ['ordered_piece', 'selected_piece']
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys='value',
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['value'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network

        :param xx: Input dict
        :return: The computed output of the network
        """
        return self.perception_net(xx)
