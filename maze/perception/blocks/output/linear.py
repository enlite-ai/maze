""" Contains a linear output head block. """
import builtins
from typing import Union, List, Sequence, Dict

import numpy as np
import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock

Number = Union[builtins.int, builtins.float, builtins.bool]


class LinearOutputBlock(ShapeNormalizationBlock):
    """A linear output head modeled as a single linear layer.
    Additionally the biases of the layer can be biased towards initial values.
    (e.g. to achieve a certain sampling behaviour right from the beginning of training)

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param output_units: Count of output units.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], output_units: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=2, out_num_dims=2)
        assert len(self.out_keys) == 1
        assert len(self.in_keys) == 1
        assert len(self.in_shapes) == 1

        self.input_units = self.in_shapes[0][-1]
        self.output_units = output_units

        self.net = nn.Linear(self.input_units, self.output_units)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """
        input_tensor = block_input[self.in_keys[0]]
        output_tensor = self.net(input_tensor)
        return {self.out_keys[0]: output_tensor}

    def set_bias(self, bias: Union[Number, np.ndarray]) -> None:
        """Reset layer biases of output head which where originally initialized with zeros.

        :param bias: The initial bias values.
        """
        if isinstance(bias, np.ndarray):
            assert self.net.bias.data.shape == bias.shape
            self.net.bias.data = torch.from_numpy(bias).to(self.net.bias.device)
        else:
            self.net.bias.data.fill_(bias)

    def __repr__(self):
        txt = self.__class__.__name__
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
