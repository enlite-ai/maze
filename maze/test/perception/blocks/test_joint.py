""" Unit tests for joint perception blocks. """
from typing import Dict

from torch import nn as nn

from maze.perception.blocks.joint_blocks.flatten_dense import FlattenDenseBlock
from maze.perception.blocks.joint_blocks.lstm_last_step import LSTMLastStepBlock
from maze.perception.blocks.joint_blocks.vgg_conv_dense import VGGConvolutionDenseBlock
from maze.perception.blocks.joint_blocks.vgg_conv_gap import VGGConvolutionGAPBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_feed_forward_conv_gap_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: VGGConvolutionGAPBlock = VGGConvolutionGAPBlock(in_keys="in_key", out_keys="out_key",
                                                         in_shapes=(3, 64, 64), hidden_channels=[4, 8, 16],
                                                         non_lin=nn.ReLU)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 16)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-1:]]


def test_feed_forward_conv_dense_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: VGGConvolutionDenseBlock = VGGConvolutionDenseBlock(in_keys="in_key", out_keys="out_key",
                                                             in_shapes=(3, 64, 64), hidden_channels=[4, 8, 16],
                                                             hidden_units=[32, 32], non_lin=nn.ReLU)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 32)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-1:]]


def test_feed_forward_flatten_dense_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: FlattenDenseBlock = FlattenDenseBlock(in_keys="in_key", out_keys="out_key",
                                               in_shapes=(3, 64, 64), num_flatten_dims=3,
                                               hidden_units=[32, 64], non_lin=nn.ReLU)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-1:]]


def test_lstm_last_step_block():
    """ perception test """
    for dims in [[32, 16], [100, 32, 16], [100, 5, 32, 16]]:
        in_dict = build_input_dict(dims=dims)

        net = LSTMLastStepBlock(in_keys="in_key", out_keys="out_key", in_shapes=(32, 16), hidden_size=64, num_layers=1,
                                bidirectional=True, non_lin=nn.ReLU)
        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        assert net.out_shapes() == [out_dict["out_key"].shape[-1:]]
        assert list(out_dict["out_key"].shape[:-1]) == dims[:-2]
        assert out_dict["out_key"].ndim == len(dims) - 1
