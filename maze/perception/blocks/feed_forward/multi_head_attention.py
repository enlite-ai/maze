"""Implementation of Multi-Head-Attention Block
"""
from typing import Union, List, Sequence, Dict, Optional

import torch
from torch import nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class MultiHeadAttentionBlock(ShapeNormalizationBlock):
    """Implementation of a torch MultiHeadAttention block.

    This Block wraps the torch.nn.MultiheadAttention. This block can then be used
    for 1d data as well as sequential data.

    :param in_keys: Keys identifying the input tensors. First key is the query, second is the key and the third input
        is the value. Additionally there is the optional attention mask which can be passed as an input to the block.
        - query: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
        the embedding dimension.
        - key: :math:`(N, S, E)`, where S is the source sequence length, N is the batch size, E is
        the embedding dimension.
        - value: :math:`(N, S, E)` where S is the source sequence length, N is the batch size, E is
        the embedding dimension.
        - (Optional) attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence
        length. 3D mask :math:`(N, L, S)` where N is the batch size, L is the target sequence length, S is the
        source sequence length. attn_mask ensure that position i is allowed to attend the unmasked positions. If a
        Bool- or int- or float-Tensor is provided, positions with ``False/~1`` is not allowed to attend while
        ``True/1`` values will be unchanged.
        - (Optional) key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
        If a Bool- or int- or float-Tensor is provided, the positions with the value of ``False/~1`` will be ignored
        while the position with the value of ``True/1`` will be unchanged.
    :param out_keys: Keys identifying the output tensors. First key is self-attention output, second optional key is
        attention map.
        - attn_output: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is the embedding
        dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size, L is the target sequence length, S is the
        source sequence length.
    :param in_shapes: List of input shapes.
    :param num_heads: Parallel attention heads.
    :param dropout: A dropout layer on attn_output_weights.
    :param bias: Add bias as module parameter.
    :param add_bias_kv: Add bias to the key and value sequences at dim=0.
    :param add_zero_attn: Add a new batch of zeros to the key and value sequences at dim=1.
    :param kdim: Total number of features in key. Default: None.
    :param vdim: Total number of features in value. Default: None.
    :param use_key_padding_mask: Specify wether a key padding mask is being used.

    Note: If kdim and vdim are None, they will be set to embed_dim such that query, key, and value have the same number
        of features.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], num_heads: int,
                 dropout: Optional[float], add_input_to_output: bool, bias: bool, add_bias_kv: bool,
                 add_zero_attn: bool, kdim: Optional[int], vdim: Optional[int], use_key_padding_mask: bool):

        in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        out_keys = out_keys if isinstance(out_keys, List) else [out_keys]
        in_shapes = in_shapes if isinstance(in_shapes, List) else [in_shapes]

        assert len(in_keys) in (3, 4), f'but got {in_keys}'
        assert len(out_keys) in (1, 2), f'but got {out_keys}'

        assert len(in_shapes[0]) in (1, 2), 'Input shape has to be 1 or 2 dimensional (without batch)'
        assert len(in_shapes[1]) in (1, 2), 'Input shape has to be 1 or 2 dimensional (without batch)'

        # Input dimensionality is inferred from the in_shapes since the block can be used with 1 or 2 dimensional data,
        #   without the batch dimension. Additionally the mask dimensions is also inferred if present.
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
        assert len(self.in_shapes[0]) in (1, 2), 'Input shape has to be 1 or 2 dimensional (without batch)'
        # Unsqueeze input if there is no sequential information, that is if the data is only 1d
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
        self.num_heads = num_heads
        self.use_key_padding_mask = use_key_padding_mask
        assert not len(self.in_keys) == 5 or self.use_key_padding_mask, 'Key padding mask has to be true if 5 ' \
                                                                        'in_keys are being used'

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               dropout=dropout if dropout is not None else 0.0,
                                               bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                               kdim=kdim, vdim=vdim)
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface"""

        # Get the tensors from the block input
        query_tensor = block_input[self.in_keys[0]]
        key_tensor = block_input[self.in_keys[1]]
        value_tensor = block_input[self.in_keys[2]]
        attn_mask = block_input[self.in_keys[3]] if len(self.in_keys) == 4 and not self.use_key_padding_mask \
            or len(self.in_keys) == 5 else None
        key_padding_mask = block_input[self.in_keys[4]] if len(self.in_keys) == 5 else block_input[self.in_keys[3]] if \
            len(self.in_keys) == 4 and self.use_key_padding_mask else None

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

        if key_padding_mask is not None:
            # We have to invert the key_padding_mask in order to work with the torch.nn.MultiheadAttention
            key_padding_mask = ~torch.eq(key_padding_mask, torch.tensor(1).to(key_padding_mask.device))

        if self.preprocess is not None:
            query_tensor = self.preprocess(query_tensor)
            key_tensor = self.preprocess(key_tensor)
            value_tensor = self.preprocess(value_tensor)

        out, attention = self.self_attn(query_tensor, key_tensor, value_tensor, need_weights=len(self.out_keys) == 2,
                                        attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self.postprocess is not None:
            out = self.postprocess(out)
            query_tensor = self.postprocess(query_tensor)

        # Scale self attention
        out = self.gamma * out

        if self.add_input_to_output:
            out = torch.add(out, query_tensor)

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
        txt += f'\n\tuse_attn_mask: ' \
               f'{len(self.in_keys) == 4 and not self.use_key_padding_mask or len(self.in_keys) == 5}'
        txt += f'\n\tuse_key_padding_mask: ' \
               f'{len(self.in_keys) == 4 and self.use_key_padding_mask or len(self.in_keys) == 5}'
        txt += f'\n\tadd_bias_kv: {self.add_bias_kv}'
        txt += f'\n\tadd_zero_attn: {self.add_zero_attn}'
        if self.kdim is not None:
            txt += f'\n\tkdim: {self.kdim}'
        if self.vdim is not None:
            txt += f'\n\tvdim: {self.vdim}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
