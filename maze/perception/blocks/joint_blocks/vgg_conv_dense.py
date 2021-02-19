""" Contains a joint vgg convolution, flattening dense perception block. """
from typing import Union, List, Sequence, Dict

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.feed_forward.vgg_conv import VGGConvolutionBlock
from maze.perception.blocks.general.flatten import FlattenBlock


class VGGConvolutionDenseBlock(PerceptionBlock):
    """A block containing multiple subsequent vgg style convolution stacks
    followed by flattening and a dense layer block.

    For details on the convolution part see :class:`~maze.perception.blocks.feed_forward.vgg_conv.VGGConvolutionBlock`.
    For details on flattening see :class:`~maze.perception.blocks.general.flatten.FlattenBlock`.
    For details on dense layers see :class:`~maze.perception.blocks.feed_forward.dense.DenseBlock`.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_channels: List containing the number of hidden channels for hidden layers.
    :param hidden_units: List containing the number of hidden units for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 hidden_channels: List[int],
                 hidden_units: List[int],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        out_keys_conv = [f"{k}_conv" for k in self.out_keys]
        self.conv_block = VGGConvolutionBlock(in_keys=in_keys, out_keys=out_keys_conv, in_shapes=in_shapes,
                                              hidden_channels=hidden_channels, non_lin=non_lin)

        out_keys_flatten = [f"{k}_flat" for k in out_keys_conv] if len(hidden_units) > 0 else out_keys
        self.flatten_block = FlattenBlock(in_keys=out_keys_conv, out_keys=out_keys_flatten,
                                          in_shapes=self.conv_block.out_shapes(), num_flatten_dims=3)

        self.dense_block = None
        if len(hidden_units) > 0:
            self.dense_block = DenseBlock(in_keys=out_keys_flatten, out_keys=out_keys,
                                          in_shapes=self.flatten_block.out_shapes(),
                                          hidden_units=hidden_units, non_lin=non_lin)

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # forward pass of submodules
        block_output = self.conv_block(block_input)
        block_output = self.flatten_block(block_output)
        if self.dense_block:
            block_output = self.dense_block(block_output)

        return block_output

    def __repr__(self):
        txt = f"{VGGConvolutionDenseBlock.__name__}:"
        txt += f"\n\n{str(self.conv_block)}"
        txt += f"\n\n{str(self.flatten_block)}"
        txt += f"\n\n{str(self.dense_block)}"
        return txt
