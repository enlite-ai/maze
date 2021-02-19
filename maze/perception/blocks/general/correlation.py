""" Contains feature correlation blocks. """
from typing import Union, List, Dict, Sequence

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class CorrelationBlock(PerceptionBlock):
    """A feature correlation block.

    This block takes two feature representation as an input and correlates them along the last dimension.
    If the blocks do not have the same number of dimensions additional 1d-dimensions are added
    to allow for broadcasting.

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param reduce: If True a sum reduction as applied along dim=-1.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], reduce: bool):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.reduce = reduce
        assert len(self.out_keys) == 1
        assert len(self.in_keys) == 2
        assert len(self.in_shapes) == 2

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # prepare input tensor
        key_tensor = block_input[self.in_keys[0]]
        query_tensor = block_input[self.in_keys[1]]
        assert key_tensor.shape[-1] == query_tensor.shape[-1]

        # insert additional dimensions for broadcasting
        max_dims = max(key_tensor.ndim, query_tensor.ndim)
        for d in range(max_dims):

            # insert broadcasting dimensions in case of dimensions mismatch
            if key_tensor.shape[d] != query_tensor.shape[d] and 1 not in [key_tensor.shape[d], query_tensor.shape[d]]:
                if key_tensor.ndim < query_tensor.ndim:
                    key_tensor = key_tensor.unsqueeze(dim=d)
                else:
                    query_tensor = query_tensor.unsqueeze(dim=d)
        assert key_tensor.ndim == query_tensor.ndim

        # compute correlation
        correlation = (key_tensor * query_tensor)

        if self.reduce:
            correlation = correlation.sum(dim=-1)

        return {self.out_keys[0]: correlation}

    def __repr__(self):
        txt = CorrelationBlock.__name__
        txt += f"\n\treduce({self.reduce})"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
