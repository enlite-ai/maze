""" Contains a global average pooling block. """
from typing import Union, List, Sequence, Dict

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class GlobalAveragePoolingBlock(ShapeNormalizationBlock):
    """A global average pooling block.
    The block expects the input tensors to have the from (batch-dim, channel-dim, row-dim, column-dim).

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=4, out_num_dims=2)

        # compile network
        self.net = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        assert input_tensor.ndim == self.in_num_dims[0]

        # forward pass
        output_tensor = self.net(input_tensor).squeeze(dim=-1).squeeze(dim=-1)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0]

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = f"{GlobalAveragePoolingBlock.__name__}"
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
