"""Test methods for the MultiheadAttention block"""
from maze.perception.blocks.feed_forward.multi_head_attention import MultiHeadAttentionBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


def test_attention_1d():
    """test_attention_1d"""
    in_dict = build_multi_input_dict(dims=[(2, 10), (2, 10), (2, 10)])

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2'],
                                              out_keys=['added_attention', 'attention'], in_shapes=[(10,), (10,),
                                                                                                    (10,)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=None, vdim=None, use_key_padding_mask=False)

    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 2
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 10)


def test_attention_sequential():
    """test_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 7, 10), (2, 7, 10), (2, 7, 10)])

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2'],
                                              out_keys='self_attention', in_shapes=[(7, 10), (7, 10), (7, 10)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=None, vdim=None, use_key_padding_mask=False)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 7, 10)


def test_attention_sequential_masked():
    """test_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 7, 10), (2, 7, 10), (2, 7, 10), (2, 7, 7)])
    in_dict['in_key_3'] = in_dict['in_key_3'] != 0

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2', 'in_key_3'],
                                              out_keys='self_attention', in_shapes=[(7, 10), (7, 10), (7, 10), (7, 7)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=None, vdim=None, use_key_padding_mask=False)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 7, 10)


def test_attention_sequential_2():
    """test_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 10), (2, 7), (2, 9)])

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2'],
                                              out_keys='self_attention', in_shapes=[(10,), (7,), (9,)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=7, vdim=9, use_key_padding_mask=False)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 371
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 10)


def test_attention_sequential_3():
    """test_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 10), (2, 10), (2, 10), (2, 1)])
    in_dict['in_key_3'] = in_dict['in_key_3'] != 0

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2', 'in_key_3'],
                                              out_keys='self_attention', in_shapes=[(10,), (10,), (10,), (1,)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=None, vdim=None, use_key_padding_mask=True)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 10)


def test_attention_sequential_masked_2():
    """test_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 20, 10), (2, 7, 5), (2, 7, 15), (2, 7)])
    in_dict['in_key_3'] = in_dict['in_key_3'] != 0

    self_attn_block = MultiHeadAttentionBlock(in_keys=['in_key_0', 'in_key_1', 'in_key_2', 'in_key_3'],
                                              out_keys='self_attention', in_shapes=[(7, 10), (7, 5), (7, 15), (7,)],
                                              num_heads=10, dropout=0.0, bias=False,
                                              add_input_to_output=True, add_bias_kv=False, add_zero_attn=False,
                                              kdim=5, vdim=15, use_key_padding_mask=True)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 20, 10)
