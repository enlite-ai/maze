""" Contains a MulitIndexSlicingBlock """
from typing import Union, List, Sequence, Dict

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class MultiIndexSlicingBlock(PerceptionBlock):
    """A multi-index-slicing block.
    This can be used rather than the short hand tensor[...,[a,b]] where [a,b] is the list given as selection indices.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param select_dim: The dimension to slice from.
    :param select_idxs: The index or indices to select.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], select_dim: int,
                 select_idxs: Union[int, Sequence[int]]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.selection_dim = select_dim
        self.selection_idxs = torch.tensor(select_idxs, requires_grad=False)
        assert self.selection_idxs.ndim == 1

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass, slicing the input tensor as defined by the selection_dim and select_idxs.

        :param block_input: The block's input dictionary.
        :return: The block's output dictionary.
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        # forward pass
        if self.selection_idxs.device != input_tensor.device:
            self.selection_idxs = self.selection_idxs.to(input_tensor.device)
        output_tensor = torch.index_select(input_tensor, dim=self.selection_dim, index=self.selection_idxs)

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f"\n\tselection_dim: {self.selection_dim}, selection_idxs: {self.selection_idxs.tolist()}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
