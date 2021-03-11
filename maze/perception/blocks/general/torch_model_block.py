""" Contains a TorchModelBlock """
from typing import Union, List, Sequence, Dict

import torch
import torch.nn as nn
from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class TorchModelBlock(ShapeNormalizationBlock):
    """A block transforming a common nn.Module to a shape-normalized Maze perception block.

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param in_num_dims: Required number of dimensions for corresponding input.
    :param out_num_dims: Required number of dimensions for corresponding output.
    :param net: An nn.Module PyTorch net (the forward method of which must accept a Tensor input dict as parameter
                and must return a Tensor output dict)
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 in_num_dims: Union[int, List[int]], out_num_dims: Union[int, List[int]], net: nn.Module,):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_num_dims,
                         out_num_dims=out_num_dims)

        self.net = net

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        for i, in_key in enumerate(self.in_keys):
            assert block_input[in_key].ndim == self.in_num_dims[i]

        # forward pass
        block_output = self.net(block_input)

        for i, out_key in enumerate(self.out_keys):
            assert block_output[out_key].ndim == self.out_num_dims[i]

        return block_output

    def __repr__(self):
        txt = f"{TorchModelBlock.__name__}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
