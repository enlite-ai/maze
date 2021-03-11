"""Implementation of Attention Block
"""
from typing import Union, List, Sequence, Dict, Optional

import torch
from torch import nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class MultiHeadAttentionBlock(ShapeNormalizationBlock):
    """Implementation of a torch MultiHeadAttention block.

    This Block wraps the torch.nn.MultiheadAttention. This block can then be used
    for 1d data as well as sequential data.

    :param in_keys: Keys identifying the input tensors. First key is self_attention output, second optional key is
        attention mask.
    :param out_keys: Keys identifying the output tensors. First key is self-attention output, second optional key is
        attention map.
    :param in_shapes: List of input shapes.
    :param num_heads: Parallel attention heads.
    :param dropout: A dropout layer on attn_output_weights.
    :param bias: Add bias as module parameter.
    :param add_bias_kv: Add bias to the key and value sequences at dim=0.
    :param add_zero_attn: Add a new batch of zeros to the key and value sequences at dim=1.
    :param kdim: Total number of features in key. Default: None.
    :param vdim: Total number of features in value. Default: None.

    Note: If kdim and vdim are None, they will be set to embed_dim such that query, key, and value have the same number
        of features.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], num_heads: int,
                 dropout: Optional[float], add_input_to_output: bool, bias: bool, add_bias_kv: bool,
                 add_zero_attn: bool, kdim: Optional[int], vdim: Optional[int]):
        in_num_dims = [3]
        if isinstance(in_keys, list) and len(in_keys) > 1:
            mask_dim = len(in_shapes[-1])
            in_num_dims.append(mask_dim + 1)
        out_num_dims = [3]
        if isinstance(out_keys, list) and len(out_keys) > 1:
            out_num_dims.append(3)
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_num_dims,
                         out_num_dims=out_num_dims)
        assert len(self.in_keys) in (2, 3)
        assert len(self.out_keys) in (1, 2)

        self.add_input_to_output = add_input_to_output
        embed_dim = self.in_shapes[0][-1]

        self.preprocess = None
        self.postprocess = None
        assert len(self.in_shapes[0]) in (1, 2), 'Input shape has to be 1 or 2 dimensional (without batch)'
        # Unsqueeze input if there is no sequential information, that if the data is only 1d
        if len(self.in_shapes[0]) == 1:
            self.preprocess = lambda x: torch.unsqueeze(x, 0)
            self.postprocess = lambda x: torch.squeeze(x, 0)
        # The batch dimension has to changed to dim 1 for the mulihead attention model
        elif len(self.in_shapes[0]) == 2:
            self.preprocess = lambda x: x.transpose(1, 0)
            self.postprocess = lambda x: x.transpose(0, 1)

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim
        self.vdim = vdim

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               dropout=dropout if dropout is not None else 0.0,
                                               bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                               kdim=kdim, vdim=vdim)
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface"""

        key_value = block_input[self.in_keys[0]]
        query = block_input[self.in_keys[1]]
        attn_mask = block_input[self.in_keys[2]] if len(self.in_keys) > 2 else None

        if attn_mask is not None:
            if self.num_heads > 1:
                attn_mask = attn_mask.repeat([self.num_heads, *[1 for _ in attn_mask.shape[1:]]])
            attn_mask = ~torch.eq(attn_mask, torch.tensor(1).to(attn_mask.device))
            attn_mask[..., 0] = False

        if self.preprocess is not None:
            key_value = self.preprocess(key_value)
            query = self.preprocess(query)

        out, attention = self.self_attn(query, key_value, key_value, need_weights=False,
                                        attn_mask=attn_mask)

        if self.postprocess is not None:
            out = self.postprocess(out)
            query = self.postprocess(query)

        if self.add_input_to_output:
            # Scale self attention
            out = self.gamma * out
            out = torch.add(out, query)

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
        txt += f'\n\tadd_bias_kv: {self.add_bias_kv}'
        txt += f'\n\tadd_zero_attn: {self.add_zero_attn}'
        if self.kdim is not None:
            txt += f'\n\tkdim: {self.kdim}'
        if self.vdim is not None:
            txt += f'\n\tvdim: {self.vdim}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
