""" Contains a joint strided convolution, flattening dense perception block. """
from typing import Union, List, Sequence, Dict, Tuple, Optional

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.feed_forward.strided_conv import StridedConvolutionBlock
from maze.perception.blocks.general.flatten import FlattenBlock


class StridedConvolutionDenseBlock(PerceptionBlock):
    """A block containing multiple subsequent strided convolutions
    followed by flattening and a dense layer block.

    For details on the convolution part see
    :class:`~maze.perception.blocks.feed_forward.strided_conv.StridedConvolutionBlock`.
    For details on flattening see :class:`~maze.perception.blocks.general.flatten.FlattenBlock`.
    For details on dense layers see :class:`~maze.perception.blocks.feed_forward.dense.DenseBlock`.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_channels: List containing the number of hidden channels for hidden layers.
    :param hidden_kernels: List containing the size of the convolving kernels.
    :param convolution_dimension: Dimension of the convolution to use [1, 2, 3]
    :param hidden_strides: List containing the strides of the convolutions.
    :param hidden_dilations: List containing the spacing between kernel elements.
    :param hidden_padding: List containing the padding added to both sides of the input
    :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
    :param hidden_units: List containing the number of hidden units for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 hidden_channels: List[int],
                 hidden_kernels: List[Union[int, Tuple[int, ...]]],
                 convolution_dimension: int,
                 hidden_strides: Optional[List[Union[int, Tuple[int, ...]]]],
                 hidden_dilations: Optional[List[Union[int, Tuple[int, ...]]]],
                 hidden_padding: Optional[List[Union[int, Tuple[int, ...]]]],
                 padding_mode: Optional[str],
                 hidden_units: List[int],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        out_keys_conv = [f"{k}_conv" for k in self.out_keys]
        self.conv_block = StridedConvolutionBlock(in_keys=in_keys, out_keys=out_keys_conv, in_shapes=in_shapes,
                                                  hidden_channels=hidden_channels,
                                                  hidden_kernels=hidden_kernels,
                                                  convolution_dimension=convolution_dimension,
                                                  hidden_strides=hidden_strides,
                                                  hidden_dilations=hidden_dilations,
                                                  hidden_padding=hidden_padding,
                                                  padding_mode=padding_mode,
                                                  non_lin=non_lin)

        out_keys_flatten = [f"{k}_flat" for k in out_keys_conv] if len(hidden_units) > 0 else out_keys
        self.flatten_block = FlattenBlock(in_keys=out_keys_conv, out_keys=out_keys_flatten,
                                          in_shapes=self.conv_block.out_shapes(), num_flatten_dims=3)

        if len(hidden_units) > 0:
            self.dense_block = DenseBlock(in_keys=out_keys_flatten, out_keys=out_keys,
                                          in_shapes=self.flatten_block.out_shapes(),
                                          hidden_units=hidden_units, non_lin=non_lin)
        else:
            self.dense_block = None

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
        txt = f"{self.__class__.__name__}:"
        txt += f"\n\n{str(self.conv_block)}"
        txt += f"\n\n{str(self.flatten_block)}"
        txt += f"\n\n{str(self.dense_block)}"
        return txt
