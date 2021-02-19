"""Cutting 2d actor networks"""
from collections import OrderedDict
from typing import Dict, Union, Sequence

import torch
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.correlation import CorrelationBlock
from maze.perception.blocks.general.functional import FunctionalBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from torch import nn as nn


class SelectionPolicyNet(nn.Module):
    """Selection Policy Network for cutting 2d.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param with_mask: Weather to use action masking or not.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], with_mask: bool):
        nn.Module.__init__(self)
        self.obs_shapes = obs_shapes

        hidden_units, embedding_dim = 32, 7

        self.perception_dict = OrderedDict()

        # embed inventory
        # ---------------
        self.perception_dict['inventory_feat'] = DenseBlock(
            in_keys='inventory', out_keys='inventory_feat', in_shapes=self.obs_shapes['inventory'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['inventory_embed'] = LinearOutputBlock(
            in_keys='inventory_feat', out_keys='inventory_embed',
            in_shapes=self.perception_dict['inventory_feat'].out_shapes(),
            output_units=embedding_dim)

        # embed ordered_piece
        # ------------------_
        self.perception_dict['order_unsqueezed'] = FunctionalBlock(
            in_keys='ordered_piece', out_keys='order_unsqueezed', in_shapes=self.obs_shapes['ordered_piece'],
            func=lambda x: torch.unsqueeze(x, dim=-2))

        self.perception_dict['order_feat'] = DenseBlock(
            in_keys='order_unsqueezed', out_keys='order_feat',
            in_shapes=self.perception_dict['order_unsqueezed'].out_shapes(),
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['order_embed'] = LinearOutputBlock(
            in_keys='order_feat', out_keys='order_embed',
            in_shapes=self.perception_dict['order_feat'].out_shapes(),
            output_units=embedding_dim)

        # compute dot product score
        # -------------------------
        in_shapes = self.perception_dict['inventory_embed'].out_shapes()
        in_shapes += self.perception_dict['order_embed'].out_shapes()
        out_key = 'corr_score' if with_mask else 'piece_idx'
        self.perception_dict[out_key] = CorrelationBlock(
            in_keys=['inventory_embed', 'order_embed'], out_keys=out_key,
            in_shapes=in_shapes, reduce=True)

        # apply action masking
        if with_mask:
            self.perception_dict['piece_idx'] = ActionMaskingBlock(
                in_keys=['corr_score', 'inventory_mask'], out_keys='piece_idx',
                in_shapes=self.perception_dict['corr_score'].out_shapes() + [self.obs_shapes['inventory_mask']],
                num_actors=1, num_of_actor_actions=None)

        assert self.perception_dict['piece_idx'].out_shapes()[0][0] == action_logits_shapes['piece_idx'][0]

        in_keys = ['ordered_piece', 'inventory']
        if with_mask:
            in_keys.append('inventory_mask')
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys='piece_idx',
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['inventory_embed'].apply(make_module_init_normc(0.01))
        self.perception_dict['order_embed'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param xx: Input dict.
        :return: The computed output of the network.
        """
        return self.perception_net(xx)


class CuttingPolicyNet(nn.Module):
    """The Policy net (actor) computing the action probabilities from the observations.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param with_mask: Weather to use action masking or not.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], with_mask: bool):
        nn.Module.__init__(self)
        self.obs_shapes = obs_shapes

        hidden_units = 32

        self.perception_dict = OrderedDict()

        self.perception_dict['selected_feat'] = DenseBlock(
            in_keys='selected_piece', out_keys='selected_feat', in_shapes=self.obs_shapes['selected_piece'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['order_feat'] = DenseBlock(
            in_keys='ordered_piece', out_keys='order_feat', in_shapes=self.obs_shapes['ordered_piece'],
            hidden_units=[hidden_units], non_lin=non_lin)

        self.perception_dict['latent'] = ConcatenationBlock(
            in_keys=['selected_feat', 'order_feat'], out_keys='latent',
            in_shapes=[[hidden_units], [hidden_units]], concat_dim=-1)

        rotation_out_key = 'cut_rotation_logits' if with_mask else 'cut_rotation'
        self.perception_dict[rotation_out_key] = LinearOutputBlock(
            in_keys='latent', out_keys=rotation_out_key, in_shapes=self.perception_dict['latent'].out_shapes(),
            output_units=action_logits_shapes['cut_rotation'][0])

        if with_mask:
            self.perception_dict['cut_rotation'] = ActionMaskingBlock(
                in_keys=['cut_rotation_logits', 'cutting_mask'], out_keys='cut_rotation',
                in_shapes=self.perception_dict['cut_rotation_logits'].out_shapes() + [self.obs_shapes['cutting_mask']],
                num_actors=1, num_of_actor_actions=None)

        self.perception_dict['cut_order'] = LinearOutputBlock(
            in_keys='latent', out_keys='cut_order', in_shapes=self.perception_dict['latent'].out_shapes(),
            output_units=action_logits_shapes['cut_order'][0])

        in_keys = ['selected_piece', 'ordered_piece']
        if with_mask:
            in_keys.append('cutting_mask')
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys=['cut_rotation', 'cut_order'],
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict[rotation_out_key].apply(make_module_init_normc(0.01))
        self.perception_dict['cut_order'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param xx: Input dict.
        :return: The computed output of the network.
        """
        return self.perception_net(xx)
