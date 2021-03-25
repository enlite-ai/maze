"""Contains tests for the point net block"""
import pytest
import torch

from maze.perception.blocks.feed_forward.point_net import PointNetFeatureTransformNet, PointNetFeatureBlock
from maze.test.perception.perception_test_utils import build_input_dict, build_multi_input_dict


def perform_test_for_parameters(num_points: int, batch_size: int, num_features: int, embedding_dim: int,
                                pooling_func_name: str, use_masking: bool) -> None:
    """Perform test on Feature transformation module with given input parameters

    :param num_points: The number of points.
    :param batch_size: The batch size.
    :param num_features: The feature dimension.
    :param embedding_dim: The embedding dimension.
    :param pooling_func_name: The pooling function string representing the function to use.
    :param use_masking: Specify whether to use masking.
    """

    pnft = PointNetFeatureTransformNet(num_features=num_features, non_lin=torch.nn.ReLU, use_batch_norm=True,
                                       embedding_dim=embedding_dim,
                                       pooling_func_name=pooling_func_name, use_masking=use_masking,
                                       num_points=num_points)
    input_tensor = torch.rand(batch_size, num_features, num_points)
    mask_tensor = torch.randint(0, 2, size=(batch_size, num_points)) if use_masking else None
    print(mask_tensor)
    assert pnft(input_tensor, mask_tensor).shape == torch.Size([batch_size, num_features, num_features])


def test_point_net_feature_transform():
    """Test transformation module with feature input"""
    num_points = 100
    batch_size = 10
    embedding_dim = 1024
    for KK in [32, 64]:
        for pooling_func_str in ['max', 'mean', 'sum']:
            perform_test_for_parameters(num_points, batch_size, KK, embedding_dim, pooling_func_str, False)

    perform_test_for_parameters(100, 10, 33, 1024, 'max', True)
    perform_test_for_parameters(100, 10, 33, 1024, 'mean', True)
    perform_test_for_parameters(100, 10, 33, 1024, 'sum', True)

    with pytest.raises(ValueError):
        perform_test_for_parameters(100, 10, 33, 1024, 'something', False)


def test_point_net_input_transformation():
    """Test transformation module with image input"""
    batch_size = 20
    num_features = 3
    num_points = 100
    pnft = PointNetFeatureTransformNet(num_features=num_features, non_lin=torch.nn.ReLU, use_batch_norm=True,
                                       embedding_dim=1024,
                                       pooling_func_name='max', use_masking=False, num_points=num_points)
    input_tensor = torch.rand(batch_size, num_features, num_points)
    assert pnft(input_tensor, None).shape == torch.Size([batch_size, num_features, num_features])


def perform_pointnet_block_test(batch_dim: int, num_points: int, num_features: int, embedding_dim: int,
                                use_batch_norm: bool, pooling_func_str: str, use_feature_transform: bool,
                                with_masking: bool):
    """Perform test on point net"""
    if not with_masking:
        in_dict = build_input_dict(dims=[batch_dim, num_points, num_features])
        in_shapes = (num_points, num_features)
    else:
        in_dict = build_multi_input_dict(dims=[[batch_dim, num_points, num_features], [batch_dim, num_points]])
        in_dict['in_key_1'] = in_dict['in_key_1'] != 0
        in_dict['in_key_1'][:, -1] = 0
        in_shapes = [(num_points, num_features), (num_points,)]

    pointnet_block = PointNetFeatureBlock(in_keys=list(in_dict.keys()), in_shapes=in_shapes, out_keys='out_key',
                                          use_batch_norm=use_batch_norm, non_lin=torch.nn.ReLU,
                                          embedding_dim=embedding_dim, pooling_func_name=pooling_func_str,
                                          use_feature_transform=use_feature_transform)
    pointnet_block.print_internal_shape_representation = True

    str(pointnet_block)
    out = pointnet_block(in_dict)['out_key']
    assert out.shape == torch.Size([batch_dim, embedding_dim])


def test_point_net_block():
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=1024, use_batch_norm=False,
                                pooling_func_str='max', use_feature_transform=True, with_masking=False)
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=True,
                                pooling_func_str='max', use_feature_transform=True, with_masking=False)

    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=True,
                                pooling_func_str='max', use_feature_transform=True, with_masking=False)
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=False,
                                pooling_func_str='max', use_feature_transform=False, with_masking=False)
    with pytest.raises(ValueError):
        perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128,
                                    use_batch_norm=False,
                                    pooling_func_str='something', use_feature_transform=False, with_masking=False)


def test_point_net_different_pooling_operations():
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=False,
                                pooling_func_str='mean', use_feature_transform=True, with_masking=False)
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=False,
                                pooling_func_str='sum', use_feature_transform=True, with_masking=False)


def test_point_net_with_masking():
    perform_pointnet_block_test(batch_dim=20, num_points=100, num_features=3, embedding_dim=128, use_batch_norm=False,
                                pooling_func_str='max', use_feature_transform=True, with_masking=True)
