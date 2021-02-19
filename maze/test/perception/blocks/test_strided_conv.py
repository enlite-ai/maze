""" Unit tests for strided convolutional perception blocks. """
from typing import Dict

from torch import nn as nn

from maze.perception.blocks.feed_forward.strided_conv import StridedConvolutionBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_strided_convolution_block_2d():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: StridedConvolutionBlock = StridedConvolutionBlock(in_keys="in_key", out_keys="out_key",
                                                           in_shapes=(3, 64, 64), hidden_channels=[4, 8, 16],
                                                           hidden_strides=[2, 2, 1], hidden_kernels=[3, 3, 5],
                                                           non_lin=nn.ReLU, convolution_dimension=2,
                                                           hidden_dilations=None, hidden_padding=None,
                                                           padding_mode='reflect')
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_channels == 16
    assert out_dict[net.out_keys[0]].shape[-3] == net.output_channels
    assert out_dict[net.out_keys[0]].shape == (100, 16, 14, 14)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-3:]]


def test_strided_convolution_block_1d():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 1, 32])
    net: StridedConvolutionBlock = StridedConvolutionBlock(in_keys="in_key", out_keys="out_key",
                                                           in_shapes=(1, 32), hidden_channels=[4],
                                                           hidden_strides=[1], hidden_kernels=[8],
                                                           non_lin=nn.ReLU, convolution_dimension=1,
                                                           hidden_dilations=[2], hidden_padding=[0],
                                                           padding_mode=None)
    str(net)
    out_dict = net(in_dict)
    params = net.get_num_of_parameters()
    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_channels == 4
    assert out_dict[net.out_keys[0]].shape[-2] == net.output_channels
    assert out_dict[net.out_keys[0]].shape == (100, 4, 18)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-2:]]


def test_strided_convolution_block_3d():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 2, 16, 16, 16])
    net: StridedConvolutionBlock = StridedConvolutionBlock(in_keys="in_key", out_keys="out_key",
                                                           in_shapes=(2, 16, 16, 16), hidden_channels=[4],
                                                           hidden_strides=[1], hidden_kernels=[4],
                                                           non_lin=nn.ReLU, convolution_dimension=3,
                                                           hidden_dilations=[1], hidden_padding=[0],
                                                           padding_mode=None)
    str(net)
    out_dict = net(in_dict)
    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_channels == 4
    assert out_dict[net.out_keys[0]].shape[-4] == net.output_channels
    assert out_dict[net.out_keys[0]].shape == (100, 4, 13, 13, 13)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-4:]]
