""" Contains general perception blocks. """
from typing import Union, List, Dict, Sequence

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class ConcatenationBlock(PerceptionBlock):
    """A feature concatenation block.

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param concat_dim: The index of the concatenation dimension.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], concat_dim: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.concat_dim = concat_dim
        assert len(self.out_keys) == 1

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # check input tensor
        input_tensors = [block_input[key] for key in self.in_keys]

        # forward pass
        output_tensor = torch.cat(input_tensors, dim=self.concat_dim)

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = ConcatenationBlock.__name__
        txt += f"\n\tconcat_dim: {self.concat_dim}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
