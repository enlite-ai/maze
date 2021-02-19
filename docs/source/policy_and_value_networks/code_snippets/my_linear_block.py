"""Contains a single linear layer block."""
import builtins
from typing import Union, List, Sequence, Dict

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock

Number = Union[builtins.int, builtins.float, builtins.bool]


class MyLinearBlock(ShapeNormalizationBlock):
    """A linear output block holding a single linear layer.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param output_units: Count of output units.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 output_units: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=2, out_num_dims=2)

        self.input_units = self.in_shapes[0][-1]
        self.output_units = output_units

        # initialize the linear layer
        self.net = nn.Linear(self.input_units, self.output_units)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """
        # extract the input tensor of the first (and here only) input key
        input_tensor = block_input[self.in_keys[0]]
        # apply the linear layer
        output_tensor = self.net(input_tensor)
        # return the output tensor as a tensor dictionary
        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        """This is the text shown in the graph visualization."""
        txt = self.__class__.__name__
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
