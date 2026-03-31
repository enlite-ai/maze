"""Contains a TorchModelBlock"""

from __future__ import annotations

from collections.abc import Sequence

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock

import torch
import torch.nn as nn


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

    def __init__(
        self,
        in_keys: str | list[str],
        out_keys: str | list[str],
        in_shapes: Sequence[int] | list[Sequence[int]],
        in_num_dims: int | list[int],
        out_num_dims: int | list[int],
        net: nn.Module,
    ):
        super().__init__(
            in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_num_dims, out_num_dims=out_num_dims
        )

        self.net = net

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface"""

        for i, in_key in enumerate(self.in_keys):
            assert block_input[in_key].ndim == self.in_num_dims[i]

        # forward pass
        block_output = self.net(block_input)

        for i, out_key in enumerate(self.out_keys):
            assert block_output[out_key].ndim == self.out_num_dims[i]

        return block_output

    def __repr__(self):
        txt = f'{TorchModelBlock.__name__}'
        txt += f'\n\tOut Shapes: {self.out_shapes()}'
        return txt
