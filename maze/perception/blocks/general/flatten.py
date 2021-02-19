""" Contains a flattening block. """
from typing import Union, List, Sequence, Dict

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class FlattenBlock(PerceptionBlock):
    """A flattening block.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param num_flatten_dims: the number of dimensions to flatten out (from right).
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], num_flatten_dims: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.num_flatten_dims = num_flatten_dims

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # prepare input tensor
        input_tensor = block_input[self.in_keys[0]]

        # forward pass
        output_tensor = torch.flatten(input_tensor, start_dim=-self.num_flatten_dims)

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = FlattenBlock.__name__
        txt += f"\n\tnum_flatten_dims: {self.num_flatten_dims}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
