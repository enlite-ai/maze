"""Dummy Base model for the dummy env"""
from typing import Dict, Sequence

import torch.nn as nn

from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.output.linear import LinearOutputBlock


class DummyBaseNet(nn.Module):
    """Maze dummy base model.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], non_lin: type(nn.Module)):
        nn.Module.__init__(self)
        self.obs_shapes = obs_shapes
        perception_dict: Dict[str, PerceptionBlock] = dict()

        for in_key, in_shape in self.obs_shapes.items():
            if len(in_shape) > 1:
                next_in_key = f'{in_key}_flat'
                perception_dict[next_in_key] = FlattenBlock(
                    in_keys=in_key, in_shapes=in_shape, out_keys=next_in_key,
                    num_flatten_dims=len(in_shape)
                )
                next_in_shape = perception_dict[next_in_key].out_shapes()
            else:
                next_in_key = in_key
                next_in_shape = in_shape

            perception_dict[f'{in_key}_encoded_feat'] = DenseBlock(
                in_keys=next_in_key, in_shapes=next_in_shape,
                out_keys=f'{in_key}_encoded_feat', non_lin=non_lin, hidden_units=[16]
            )
            perception_dict[f'{in_key}_encoded_layer'] = LinearOutputBlock(
                in_keys=f'{in_key}_encoded_feat', in_shapes=perception_dict[f'{in_key}_encoded_feat'].out_shapes(),
                out_keys=f'{in_key}_encoded_layer', output_units=8
            )

        concat_in_keys = [key for key in perception_dict.keys() if '_encoded_layer' in key]
        perception_dict['hidden_out'] = ConcatenationBlock(
            in_keys=concat_in_keys,
            in_shapes=sum([perception_dict[key].out_shapes() for key in concat_in_keys], []),
            out_keys='hidden_out', concat_dim=-1
        )

        self.perception_dict = perception_dict
