"""Implementation of attention blocks

2d data: self attention with multi-head possible from pytorch
         - build in is already the linear projection
3d data: self attention needs to use 1d convolutions to project the data to the embedding dims

"""
from typing import Union, List, Sequence, Dict, Optional

import torch
from torch import nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class SelfAttentionSeqBlock(ShapeNormalizationBlock):
    """Implementation of a self-attention block as described by reference: https://arxiv.org/abs/1706.03762

    Within this block the torch nn.MuliheadAttention is used to model the self attention. This block can then be used
    for 1d data as well as sequential data, where the embedding dimensionality has to be equal to the last dimension
    of the input.

    :param in_keys: Keys identifying the input tensors. First key is self_attention output, second optional key is
        attention mask.
    :param out_keys: Keys identifying the output tensors. First key is self-attention output, second optional key is
        attention map.
    :param in_shapes: List of input shapes.
    :param num_heads: Parallel attention heads.
    :param dropout: A dropout layer on attn_output_weights.
    :param add_input_to_output: Specifies weather the computed self attention is added to the input and returned.
    :param bias: Add bias as module parameter.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], num_heads: int,
                 dropout: Optional[float], add_input_to_output: bool, bias: bool):

        in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        out_keys = out_keys if isinstance(out_keys, List) else [out_keys]
        in_shapes = in_shapes if isinstance(in_shapes, List) else [in_shapes]

        assert isinstance(in_keys, str) or len(in_keys) in (1, 2), f'but got {in_keys}'
        assert isinstance(out_keys, str) or len(out_keys) in (1, 2), f'but got {out_keys}'

        assert len(in_shapes[0]) in (1, 2), 'Input shape has to be 1 or 2 dimensional (without batch)'

        # Input dimensionality is inferred from the in_shapes since the block can be used with 1 or 2 dimensional data,
        #   without the batch dimension. Additionally the mask dimension is also inferred if present.
        in_num_dims = [len(in_shape) + 1 for in_shape in in_shapes]

        # Output dimensionality is equal to the input dimensionality, while the dimensionality of the attention weights
        #  is only added if the a second out_key is given.
        out_num_dims = [in_num_dims[0]]
        if isinstance(out_keys, list) and len(out_keys) > 1:
            out_num_dims.append(out_num_dims[0])
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_num_dims,
                         out_num_dims=out_num_dims)

        embed_dim = self.in_shapes[0][-1]
        self.add_input_to_output = add_input_to_output

        self.preprocess = None
        self.postprocess = None
        # Unsqueeze input if there is no sequential information, that if the data is only 1d
        if len(self.in_shapes[0]) == 1:
            self.preprocess = lambda x: torch.unsqueeze(x, 0)
            self.postprocess = lambda x: torch.squeeze(x, 0)
        # The batch dimension has to changed to dim 1 for the mulihead attention model
        elif len(self.in_shapes[0]) == 2:
            self.preprocess = lambda x: x.transpose(1, 0)
            self.postprocess = lambda x: x.transpose(0, 1)

        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               dropout=dropout if dropout is not None else 0.0,
                                               bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface"""

        # Get the tensors from the block input
        input_tensor = block_input[self.in_keys[0]]
        attn_mask = block_input[self.in_keys[1]] if len(self.in_keys) > 1 else None

        # If the attention mask is used, it has to have the shape (num_heads * batch_size, target sequence length,
        #   source sequence length). Thus we have to repeat for the number of heads.
        if attn_mask is not None:
            if self.num_heads > 1:
                attn_mask = attn_mask.repeat([self.num_heads, *[1 for _ in attn_mask.shape[1:]]])
            # Furthermore we have to invert the mask in order to work with the torch.nn.MultiheadAttention
            attn_mask = ~torch.eq(attn_mask, torch.tensor(1).to(attn_mask.device))
            # Finally the first value of the mask is set to true in oder to circumvent nan values, while still ensuring
            #   fast processing of the block.
            attn_mask[..., 0] = False

        if self.preprocess is not None:
            input_tensor = self.preprocess(input_tensor)

        out, attention = self.self_attn(input_tensor, input_tensor, input_tensor, need_weights=len(self.out_keys) == 2,
                                        attn_mask=attn_mask)

        if self.postprocess is not None:
            out = self.postprocess(out)
            input_tensor = self.postprocess(input_tensor)

        # Scale self attention
        out = self.gamma * out

        if self.add_input_to_output:
            out = out + input_tensor

        out_dict = dict({self.out_keys[0]: out})
        # If two out keys are given, the second one is the attention
        if len(self.out_keys) == 2:
            out_dict[self.out_keys[1]] = attention
        return out_dict

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f'\n\tnum_heads: {self.self_attn.num_heads}'
        txt += f'\n\tembed_dim: {self.self_attn.embed_dim}'
        txt += f'\n\tdropout: {self.self_attn.dropout}'
        txt += f'\n\tbias: {self.self_attn.in_proj_bias is not None}'
        txt += f'\n\tadd_input_to_output: {self.add_input_to_output}'
        txt += f'\n\tuse_attn_mask: {len(self.in_keys) > 1}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
