"""Test methods for the self attention block"""
from maze.perception.blocks.general.self_attention_conv import SelfAttentionConvBlock
from maze.perception.blocks.general.self_attention_seq import SelfAttentionSeqBlock
from maze.test.perception.perception_test_utils import build_input_dict, build_multi_input_dict


def test_self_attention_1d():
    """test_self_attention_1d"""
    in_dict = build_input_dict(dims=(2, 10))

    self_attn_block = SelfAttentionSeqBlock(in_keys='in_key', out_keys=['self_attention', 'attention'], in_shapes=(10,),
                                            num_heads=10, dropout=0.0, bias=False,
                                            add_input_to_output=True)

    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 2
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 10)


def test_self_attention_sequential():
    """test_self_attention_sequential"""
    in_dict = build_input_dict(dims=(2, 7, 10))

    self_attn_block = SelfAttentionSeqBlock(in_keys='in_key', out_keys='self_attention', in_shapes=(7, 10),
                                            num_heads=10, dropout=0.0, bias=False,
                                            add_input_to_output=False)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 7, 10)


def test_self_attention_sequential_masked():
    """test_self_attention_sequential"""
    in_dict = build_multi_input_dict(dims=[(2, 7, 10), (2, 7, 7)])
    in_dict['in_key_1'] = in_dict['in_key_1'] != 0

    self_attn_block = SelfAttentionSeqBlock(in_keys=['in_key_0', 'in_key_1'],
                                            out_keys='self_attention', in_shapes=[(7, 10), (7, 7)],
                                            num_heads=10, dropout=0.0, bias=False,
                                            add_input_to_output=False)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert self_attn_block.get_num_of_parameters() == 411
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 1
    assert out_dict[self_attn_block.out_keys[0]].shape == (2, 7, 10)


def test_self_attention_2d():
    """test_self_attention_2d"""
    in_dict = build_multi_input_dict(dims=[(2, 16, 5, 5), (2, 25, 25)])
    in_dict['in_key_1'] = in_dict['in_key_1'] != 0

    self_attn_block = SelfAttentionConvBlock(in_keys=['in_key_0', 'in_key_1'], out_keys=['self_attention', 'attention'],
                                             in_shapes=[(16, 5, 5), (25, 25)], embed_dim=2, add_input_to_output=True,
                                             bias=True, dropout=None)
    str(self_attn_block)
    out_dict = self_attn_block(in_dict)
    assert len(out_dict.keys()) == len(self_attn_block.out_keys) == 2
    assert out_dict[self_attn_block.out_keys[0]].shape == in_dict['in_key_0'].shape
    assert out_dict[self_attn_block.out_keys[1]].shape == (2, 25, 25)
