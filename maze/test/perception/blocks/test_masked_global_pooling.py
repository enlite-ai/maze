""" Unit tests for masked global pooling perception blocks. """
from typing import Dict

import pytest
from pytest import raises

from maze.perception.blocks.general.masked_global_pooling import MaskedGlobalPoolingBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


def perform_masked_global_pooling_test_dim2_pool_1(feature_dim_1: int, feature_dim_2: int, use_masking: bool,
                                       pooling_func_name: str):
    batch_dim = 4
    in_dict = build_multi_input_dict(dims=[[batch_dim, feature_dim_1, feature_dim_2], [batch_dim, feature_dim_1]])

    net: MaskedGlobalPoolingBlock = MaskedGlobalPoolingBlock(
        in_keys=["in_key_0"] if not use_masking else ['in_key_0', 'in_key_1'],
        out_keys="out_key",
        in_shapes=[(feature_dim_1, feature_dim_2)]
        if not use_masking else [(feature_dim_1, feature_dim_2), (feature_dim_1,)],
        pooling_func=pooling_func_name, pooling_dim=-2)

    out_dict = net(in_dict if use_masking else {'in_key_0': in_dict['in_key_0']})
    str(net)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (batch_dim, feature_dim_2)


def perform_masked_global_pooling_test_dim3_pool_2(feature_dim_1: int, feature_dim_2: int, feature_dim_3: int,
                                            use_masking: bool, pooling_func_name: str):
    batch_dim = 4
    in_dict = build_multi_input_dict(dims=[[batch_dim, feature_dim_1, feature_dim_2, feature_dim_3],
                                           [batch_dim, feature_dim_1, feature_dim_2]])

    net: MaskedGlobalPoolingBlock = MaskedGlobalPoolingBlock(
        in_keys=["in_key_0"] if not use_masking else ['in_key_0', 'in_key_1'],
        out_keys="out_key",
        in_shapes=[(feature_dim_1, feature_dim_2, feature_dim_3)]
        if not use_masking else [(feature_dim_1, feature_dim_2, feature_dim_3), (feature_dim_1, feature_dim_2)],
        pooling_func=pooling_func_name, pooling_dim=-2)

    out_dict = net(in_dict if use_masking else {'in_key_0': in_dict['in_key_0']})
    str(net)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (batch_dim, feature_dim_1, feature_dim_3)

def perform_masked_global_pooling_test_dim3_pool_3(feature_dim_1: int, feature_dim_2: int, feature_dim_3: int,
                                            use_masking: bool, pooling_func_name: str):
    batch_dim = 4
    in_dict = build_multi_input_dict(dims=[[batch_dim, feature_dim_1, feature_dim_2, feature_dim_3],
                                           [batch_dim, feature_dim_1, feature_dim_2]])

    net: MaskedGlobalPoolingBlock = MaskedGlobalPoolingBlock(
        in_keys=["in_key_0"] if not use_masking else ['in_key_0', 'in_key_1'],
        out_keys="out_key",
        in_shapes=[(feature_dim_1, feature_dim_2, feature_dim_3)]
        if not use_masking else [(feature_dim_1, feature_dim_2, feature_dim_3), (feature_dim_1, feature_dim_2)],
        pooling_func=pooling_func_name, pooling_dim=-1)

    out_dict = net(in_dict if use_masking else {'in_key_0': in_dict['in_key_0']})
    str(net)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (batch_dim, feature_dim_1, feature_dim_2)


def test_masked_global_pooling_avg_block_2d_data_dim2_pool2():
    """ perception test """
    fd1 = 5
    fd2 = 3

    for pooling_func_name in ['mean', 'sum', 'max']:
        for use_masking in [True, False]:
            perform_masked_global_pooling_test_dim2_pool_1(fd1, fd2, use_masking, pooling_func_name)


def test_masked_global_pooling_avg_block_2d_data_dim3_pool3():
    """ perception test """
    fd1 = 5
    fd2 = 3
    fd3 = 7

    for pooling_func_name in ['mean', 'sum', 'max']:
        for use_masking in [True, False]:
            perform_masked_global_pooling_test_dim3_pool_3(fd1, fd2, fd3, use_masking, pooling_func_name)


def test_masked_global_pooling_avg_block_2d_data_dim3_pool2():
    """ perception test """
    fd1 = 5
    fd2 = 3
    fd3 = 7

    for pooling_func_name in ['mean', 'sum', 'max']:
        for use_masking in [True, False]:
            perform_masked_global_pooling_test_dim3_pool_2(fd1, fd2, fd3, use_masking, pooling_func_name)


def test_not_allowed_case():
    with pytest.raises(ValueError):
        perform_masked_global_pooling_test_dim3_pool_2(3, 5, 7, True, 'something')

