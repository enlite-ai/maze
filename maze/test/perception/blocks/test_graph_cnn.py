"""Test methods for the graph convolutional layer and block"""
from typing import Tuple, Dict

import pytest
import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.graph_conv import GraphConvBlock, GraphConvLayer
from maze.perception.weight_init import make_module_init_normc
from maze.test.perception.perception_test_utils import build_input_dict


def test_adj_matrix_construction_wrong_input_format():
    """Test the construction of the adj_hat matrix"""
    adj = torch.tensor([[0, 1, 1],
                        [0, 0, 0]])

    with pytest.raises(AssertionError):
        _ = GraphConvBlock.preprocess_adj_to_adj_hat(adj)

    adj = torch.tensor([[0, 1, 1],
                        [0, 0, 0],
                        [0, 0, 0]])

    with pytest.raises(AssertionError):
        _ = GraphConvBlock.preprocess_adj_to_adj_hat(adj)


def construct_pre_processing_matrix() -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct the test input, and output matrix

    :return: Return the input and output matrix for adjacency matrix pre-processing
    """
    in_matrix = torch.tensor([[0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 0, 1, 1],
                              [0, 1, 1, 0, 1],
                              [1, 1, 1, 1, 0]])

    result_matrix = torch.tensor([[0.5000, 0.0000, 0.0000, 0.0000, 0.3162],
                                  [0.0000, 0.3333, 0.0000, 0.2887, 0.2582],
                                  [0.0000, 0.0000, 0.3333, 0.2887, 0.2582],
                                  [0.0000, 0.2887, 0.2887, 0.2500, 0.2236],
                                  [0.3162, 0.2582, 0.2582, 0.2236, 0.2000]])
    return in_matrix, result_matrix


def test_adj_matrix_construction():
    """Test the construction of the adj_hat matrix"""
    in_matrix, result_matrix = construct_pre_processing_matrix()

    adj_hat_torch = GraphConvBlock.preprocess_adj_to_adj_hat(in_matrix)
    assert isinstance(adj_hat_torch, torch.Tensor)
    assert torch.allclose(adj_hat_torch, result_matrix.to(torch.float32), rtol=1.e-4)


def test_gradient_computation():
    """Test the gradient flow """
    in_matrix, result_matrix = construct_pre_processing_matrix()

    adj_hat_torch = GraphConvBlock.preprocess_adj_to_adj_hat(in_matrix)
    assert adj_hat_torch.requires_grad is False

    self_scaling_param = torch.tensor(1.0, requires_grad=False)
    adj_hat_torch = GraphConvBlock.preprocess_adj_to_adj_hat(in_matrix, self_scaling_param)
    assert adj_hat_torch.requires_grad is False

    self_scaling_param = torch.tensor(1.0, requires_grad=True)
    adj_hat_torch = GraphConvBlock.preprocess_adj_to_adj_hat(in_matrix, self_scaling_param)
    assert adj_hat_torch.requires_grad is True
    loss = sum(sum(adj_hat_torch - result_matrix))
    loss.backward()
    assert torch.isclose(self_scaling_param.grad, torch.tensor(0.1078), rtol=1.e-3)

    self_scaling_param = torch.tensor(1.0, requires_grad=True)
    with torch.set_grad_enabled(False):
        adj_hat_torch = GraphConvBlock.preprocess_adj_to_adj_hat(in_matrix, self_scaling_param)
    assert adj_hat_torch.requires_grad is False


def test_graph_cnn_layer():
    """Test the graph conv layer"""

    feat_input = build_input_dict(dims=[100, 5, 7])['in_key']
    adj_matrix_batch = construct_pre_processing_matrix()[0].repeat([100, 1, 1]).to(torch.float32)
    graph_conv_layer = GraphConvLayer(in_features=7, out_features=11, bias=False)
    assert graph_conv_layer.weight.requires_grad is True
    assert graph_conv_layer.weight.shape == torch.Size([7, 11])
    assert graph_conv_layer.bias is None
    str(graph_conv_layer)
    out = graph_conv_layer(feat_input, adj_matrix_batch)
    assert out.shape == torch.Size([100, 5, 11])

    graph_conv_layer = GraphConvLayer(in_features=7, out_features=11, bias=True)
    assert graph_conv_layer.weight.requires_grad is True
    assert graph_conv_layer.weight.shape == torch.Size([7, 11])
    assert graph_conv_layer.bias is not None
    assert graph_conv_layer.bias.requires_grad is True
    assert graph_conv_layer.bias.shape == torch.Size([11])
    out = graph_conv_layer(feat_input, adj_matrix_batch)
    assert out.shape == torch.Size([100, 5, 11])
    str(graph_conv_layer)


def test_graph_cnn_block():
    """Test the graph conv block"""

    in_dict = build_input_dict(dims=[100, 5, 7])
    adj_matrix_batch = construct_pre_processing_matrix()[0].repeat([100, 1, 1])
    in_dict['adj_matrix'] = adj_matrix_batch
    net: GraphConvBlock = GraphConvBlock(in_keys=list(in_dict.keys()), out_keys='out_key',
                                         in_shapes=[value.shape[1:] for value in in_dict.values()],
                                         hidden_features=[11, 13], bias=[False, True], non_lins=[nn.ReLU, nn.Identity],
                                         node_self_importance=1.0, trainable_node_self_importance=False,
                                         preprocess_adj=True)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_features == 13
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_features
    assert out_dict[net.out_keys[0]].shape == (100, 5, 13)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-2:]]
    assert net.get_num_of_parameters() == (7 * 11) + (11 * 13) + 13

    net.apply(make_module_init_normc(1.0))


def test_graph_cnn_block_trainable_self_importance():
    """Test the graph conv block with a trainable self importance scalar"""

    in_dict = build_input_dict(dims=[100, 5, 7])
    adj_matrix_batch = construct_pre_processing_matrix()[0].repeat([100, 1, 1])
    in_dict['adj_matrix'] = adj_matrix_batch
    net: GraphConvBlock = GraphConvBlock(in_keys=list(in_dict.keys()), out_keys='out_key',
                                         in_shapes=[value.shape[1:] for value in in_dict.values()],
                                         hidden_features=[11, 13], bias=[False, True],
                                         non_lins=[nn.ReLU, 'torch.nn.Identity'],
                                         node_self_importance=1.0, trainable_node_self_importance=True,
                                         preprocess_adj=True)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_features == 13
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_features
    assert out_dict[net.out_keys[0]].shape == (100, 5, 13)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-2:]]
    assert net.get_num_of_parameters() == (7 * 11) + (11 * 13) + 13 + 1

    net.apply(make_module_init_normc(1.0))
