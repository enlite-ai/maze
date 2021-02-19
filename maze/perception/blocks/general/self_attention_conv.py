"""Implementation of attention blocks

2d data: self attention with multi-head possible from pytorch
         - build in is already the linear projection
3d data: self attention needs to use 1d convolutions to project the data to the embedding dims
        - not difficult to implement without mask and stuff
        - can/should we do mulihead here as well?

"""
from typing import Union, List, Sequence, Dict, Optional

import numpy as np
import torch
from torch import nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class SelfAttentionConvBlock(PerceptionBlock):
    """Implementation of a self-attention block as described by reference: https://arxiv.org/abs/1805.08318

    This block can then be used for 2d data (images), to compute the self attention. If two out_keys are given, the
    actual attention is returned from the forward pass with the second out_key. Otherwise only the computed
    self-attention is returned

    :param in_keys: Keys identifying the input tensors. First key is self_attention output, second optional key is
        attention mask.
    :param out_keys: Keys identifying the output tensors. First key is self-attention output, second optional key is
        attention map.
    :param in_shapes: List of input shapes.
    :param embed_dim: The embedding dimensionality, which should be an even fraction of the input channels.
    :param add_input_to_output: Specifies weather the computed self attention is added to the input and returned.
    :param bias: Specify weather to use a bias in the projections.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 embed_dim: int, dropout: Optional[float], add_input_to_output: bool, bias: bool):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        # Assertions
        assert len(self.in_keys) == len(self.in_shapes)
        assert len(self.in_keys) in (1, 2)
        # If two out keys are given, the seconds one is the attention
        assert len(self.out_keys) in (1, 2)
        assert len(self.in_shapes[0]) == 3, 'In dimensionality should be 3 without batch'
        if len(self.in_keys) > 1:
            assert len(self.in_shapes[1]) == 2
            assert np.prod(self.in_shapes[0][-2:]) == self.in_shapes[1][-1] == self.in_shapes[1][-2]
        in_dim = self.in_shapes[0][0]
        assert in_dim > embed_dim
        assert in_dim // embed_dim == in_dim / float(embed_dim), 'in_dim should be evenly dividable by embed_dim'

        self.in_dim = in_dim
        self.embedding_dim = embed_dim
        self.add_input_to_output = add_input_to_output

        self.query_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=self.embedding_dim, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=self.embedding_dim, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.0)

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface"""

        # input_tensor.shape = (B X C X W X H)
        input_tensor = block_input[self.in_keys[0]]
        attn_mask = block_input[self.in_keys[1]] if len(self.in_keys) > 1 else None

        assert len(input_tensor.shape) == 4
        batch_size, num_channels, width, height = input_tensor.size()

        # N = width * height, E = embed_dim
        proj_query = self.query_conv(input_tensor).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X E X (N)
        proj_key = self.key_conv(input_tensor).view(batch_size, -1, width * height)  # B X E x (N)
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        if attn_mask is not None:
            energy += attn_mask

        attention = self.softmax(energy)  # B X (N) X (N)
        attention = self.dropout(attention)

        proj_value = self.value_conv(input_tensor).view(batch_size, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_channels, width, height)

        out = self.gamma * out
        if self.add_input_to_output:
            out = out + input_tensor

        out_dict = dict({self.out_keys[0]: out})
        # If two out keys are given, the seconds one is the attention
        if len(self.out_keys) == 2:
            out_dict[self.out_keys[1]] = attention
        return out_dict

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f'\n\tembed_dim: {self.embedding_dim}'
        txt += f'\n\tdropout: {self.dropout}'
        txt += f'\n\tbias: {self.query_conv.bias is not None}'
        txt += f'\n\tadd_input_to_output: {self.add_input_to_output}'
        txt += f'\n\tuse_attn_mask: {len(self.in_keys) > 1}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
