""" Unit tests for masked global pooling perception blocks. """
from typing import Dict

from pytest import raises

from maze.perception.blocks.general.masked_global_pooling import MaskedGlobalPoolingBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


def test_masked_global_pooling_avg_block():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[4, 3, 2], [4, 3]])

    # make tensor binary
    in_dict["in_key_1"] = in_dict["in_key_1"] > 0

    net: MaskedGlobalPoolingBlock = MaskedGlobalPoolingBlock(
        in_keys=["in_key_0", 'in_key_1'], out_keys="out_key", in_shapes=[(4, 3, 2), (4, 3)],
        pooling_func="mean", pooling_dim=-2)

    out_dict = net(in_dict)
    str(net)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (4, 2)

    net: MaskedGlobalPoolingBlock = MaskedGlobalPoolingBlock(
        in_keys=["in_key_0", 'in_key_1'], out_keys="out_key", in_shapes=[(4, 3, 2), (4, 3)],
        pooling_func="mean", pooling_dim=-1)

    out_dict = net(in_dict)
    str(net)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (4, 3)

    # test for invalid pooling functions
    with raises(ValueError):
        MaskedGlobalPoolingBlock(
            in_keys=["in_key_0", 'in_key_1'], out_keys="out_key", in_shapes=[(4, 3, 2), (4, 3)],
            pooling_func="my_pooling_function", pooling_dim=-1)
