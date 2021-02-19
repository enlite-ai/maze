""" Unit tests for functional perception blocks. """
from typing import Dict

import torch

from maze.perception.blocks.general.functional import FunctionalBlock
from maze.test.perception.perception_test_utils import build_input_dict, build_multi_input_dict


def test_functional_block_single_arg():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 64, 1])
    net: FunctionalBlock = FunctionalBlock(in_keys="in_key", out_keys="out_key",
                                           in_shapes=(100, 64, 1), func=torch.squeeze)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64)


def test_functional_block_single_arg_lambda():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 64, 1])
    net: FunctionalBlock = FunctionalBlock(in_keys="in_key", out_keys="out_key",
                                           in_shapes=(100, 64, 1), func=lambda value: torch.squeeze(value))
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64)


def test_functional_block_single_arg_lambda_unsqueeze():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 64])
    net: FunctionalBlock = FunctionalBlock(in_keys="in_key", out_keys="out_key",
                                           in_shapes=(100, 64), func=lambda value: torch.unsqueeze(value, dim=-1))
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64, 1)


def test_functional_block_multi_arg():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[100, 64, 1], [100, 64, 1]])

    def my_func(in_key_0, in_key_1):
        return torch.cat((in_key_0, in_key_1), dim=-1)

    net: FunctionalBlock = FunctionalBlock(in_keys=["in_key_0", 'in_key_1'], out_keys="out_key",
                                           in_shapes=[(100, 64, 1), (100, 64, 1)],
                                           func=my_func)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64, 2)


def test_functional_block_multi_arg_lambda():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[100, 64, 1], [100, 64, 1]])

    net: FunctionalBlock = FunctionalBlock(in_keys=["in_key_0", 'in_key_1'], out_keys="out_key",
                                           in_shapes=[(100, 64, 1), (100, 64, 1)],
                                           func=lambda in_key_0, in_key_1: torch.cat((in_key_0, in_key_1), dim=-1))
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64, 2)


def test_functional_block_multi_arg_order():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[100, 64], [100, 64, 1]])

    def my_func(in_key_1, in_key_0):
        squeeze_in_1 = torch.squeeze(in_key_1, dim=-1)
        return torch.cat((in_key_0, squeeze_in_1), dim=-1)

    net: FunctionalBlock = FunctionalBlock(in_keys=["in_key_0", 'in_key_1'], out_keys="out_key",
                                           in_shapes=[(100, 64), (100, 64, 1)],
                                           func=my_func)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 128)


def test_functional_block_multi_arg_multi_out():
    """ perception test """
    in_dict = build_multi_input_dict(dims=[[100, 64, 32, 1], [100, 64, 1]])

    def my_func(in_key_1, in_key_0):
        return torch.squeeze(in_key_0), torch.squeeze(in_key_1)

    net: FunctionalBlock = FunctionalBlock(in_keys=["in_key_0", 'in_key_1'], out_keys=["out_key_0", 'out_key_1'],
                                           in_shapes=[(100, 64, 32, 1), (100, 64, 1)],
                                           func=my_func)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 64, 32)
    assert out_dict[net.out_keys[1]].shape == (100, 64)
