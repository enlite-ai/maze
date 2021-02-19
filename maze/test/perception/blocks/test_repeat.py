""" Unit tests for repeat perception blocks. """
from typing import Dict

import torch

from maze.perception.blocks.general.repeat_to_match import RepeatToMatchBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


def test_repeat_block():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[4, 1, 2], [4, 100, 2]])
    net: RepeatToMatchBlock = RepeatToMatchBlock(
        in_keys=["in_key_0", 'in_key_1'], out_keys="out_key", in_shapes=[(4, 1, 2), (4, 100, 2)],
        repeat_at_idx=-2)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (4, 100, 2)
    for i in range(out_dict[net.out_keys[0]].shape[0]):
        for j in range(out_dict[net.out_keys[0]].shape[1]):
            assert all(torch.eq(out_dict[net.out_keys[0]][i][j], in_dict['in_key_0'][i][0]))


def test_repeat_block_list_out_keys():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[4, 1, 2], [4, 100, 2]])
    net: RepeatToMatchBlock = RepeatToMatchBlock(
        in_keys=["in_key_0", 'in_key_1'], out_keys=["out_key"], in_shapes=[(4, 1, 2), (4, 100, 2)],
        repeat_at_idx=-2)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (4, 100, 2)
    for i in range(out_dict[net.out_keys[0]].shape[0]):
        for j in range(out_dict[net.out_keys[0]].shape[1]):
            assert all(torch.eq(out_dict[net.out_keys[0]][i][j], in_dict['in_key_0'][i][0]))
