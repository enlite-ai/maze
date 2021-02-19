""" Contains a slice block. """
from typing import Union, List, Sequence, Dict

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class SliceBlock(PerceptionBlock):
    """A slicing block. This is for example useful for slicing the last time step in recurrent blocks by its index.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param slice_dim: The dimension to slice from.
    :param slice_idx: The index to slice.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], slice_dim: int, slice_idx: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.slice_dim = slice_dim
        self.slice_idx = slice_idx

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]

        # forward pass
        output_tensor = torch.narrow(input_tensor, dim=self.slice_dim, start=self.slice_idx, length=1)
        output_tensor = output_tensor.squeeze(dim=self.slice_dim)

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = f"{SliceBlock.__name__}"
        txt += f"\nslice_dim: {self.slice_dim}, slice_idx: {self.slice_idx}"
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
