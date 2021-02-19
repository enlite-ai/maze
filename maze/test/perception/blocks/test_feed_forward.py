""" Unit tests for feed forward perception blocks. """
from typing import Dict

from torch import nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_feed_forward_dense_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 16])
    net: DenseBlock = DenseBlock(in_keys="in_key", out_keys="out_key", in_shapes=(16,), hidden_units=[32, 32],
                                 non_lin=nn.ReLU)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_units == 32
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_units
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-1:]]


def test_feed_forward_dense_block_shape_normalization():
    """ perception test """

    for dims in [[16], [100, 16], [100, 5, 16]]:

        in_dict = build_input_dict(dims=dims)
        net = DenseBlock(in_keys="in_key", out_keys="out_key", in_shapes=(16,), hidden_units=[32, 32], non_lin=nn.ReLU)
        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        for out_key in net.out_keys:
            assert out_dict[out_key].shape[-1] == net.output_units
