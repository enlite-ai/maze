""" Unit tests for convolutional perception blocks. """
from typing import Dict

from torch import nn as nn

from maze.perception.blocks.feed_forward.vgg_conv import VGGConvolutionBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_feed_forward_convolution_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: VGGConvolutionBlock = VGGConvolutionBlock(in_keys="in_key", out_keys="out_key",
                                                   in_shapes=(3, 64, 64), hidden_channels=[4, 8, 16], non_lin=nn.ReLU)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_channels == 16
    assert out_dict[net.out_keys[0]].shape[-3] == net.output_channels
    assert out_dict[net.out_keys[0]].shape == (100, 16, 8, 8)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-3:]]
