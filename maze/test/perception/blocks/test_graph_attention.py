"""Test the graph attention block and layer"""
from typing import Dict, Tuple

import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.graph_attention import GraphAttentionBlock, GraphAttentionLayer, \
    GraphMultiHeadAttentionLayer
from maze.perception.weight_init import make_module_init_normc
from maze.test.perception.perception_test_utils import build_input_dict


def construct_pre_processing_matrix_adj_bar() -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct the test input, and output matrix

    :return: Return the input and output matrix for adjacency matrix pre-processing
    """
    in_matrix = torch.tensor([[0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 0, 1, 1],
                              [0, 1, 1, 0, 1],
                              [1, 1, 1, 1, 0]])

    result_matrix = in_matrix + torch.eye(in_matrix.shape[0])

    return in_matrix, result_matrix


def test_adj_matrix_construction():
    """Test the construction of the adj_hat matrix"""
    in_matrix, result_matrix = construct_pre_processing_matrix_adj_bar()

    adj_hat_torch = GraphAttentionBlock.preprocess_adj_to_adj_bar(in_matrix)
    assert isinstance(adj_hat_torch, torch.Tensor)
    assert torch.allclose(adj_hat_torch, result_matrix.to(torch.float32), rtol=1.e-4)


def test_graph_attention_layer():
    """Test the graph conv layer"""

    feat_input = build_input_dict(dims=[100, 5, 7])['in_key']
    adj_matrix_batch = construct_pre_processing_matrix_adj_bar()[0].repeat([100, 1, 1]).to(torch.float32)
    graph_conv_layer = GraphAttentionLayer(in_features=7, out_features=11, alpha=0.2, dropout=0.0)
    assert graph_conv_layer.weight.requires_grad is True
    assert graph_conv_layer.weight.shape == torch.Size([7, 11])
    str(graph_conv_layer)
    out = graph_conv_layer(feat_input, adj_matrix_batch)
    assert out.shape == torch.Size([100, 5, 11])

    graph_conv_layer = GraphAttentionLayer(in_features=7, out_features=11, alpha=.2, dropout=0.0)
    assert graph_conv_layer.weight.requires_grad is True
    assert graph_conv_layer.weight.shape == torch.Size([7, 11])
    out = graph_conv_layer(feat_input, adj_matrix_batch)
    assert out.shape == torch.Size([100, 5, 11])
    str(graph_conv_layer)


def test_graph_multi_head_layer():
    """Test the graph conv layer"""

    feat_input = build_input_dict(dims=[100, 5, 7])['in_key']
    adj_matrix_batch = construct_pre_processing_matrix_adj_bar()[0].repeat([100, 1, 1]).to(torch.float32)
    graph_conv_layer = GraphMultiHeadAttentionLayer(in_features=7, out_features=11, alpha=0.2, n_heads=3, dropout=0,
                                                    avg_out=False)
    str(graph_conv_layer)
    out = graph_conv_layer(feat_input, adj_matrix_batch)
    assert out.shape == torch.Size([100, 5, 33])


def test_graph_attention_block():
    """Test the graph conv block"""

    in_dict = build_input_dict(dims=[100, 5, 7])
    adj_matrix_batch = construct_pre_processing_matrix_adj_bar()[0].repeat([100, 1, 1])
    in_dict['adj_matrix'] = adj_matrix_batch
    net: GraphAttentionBlock = GraphAttentionBlock(in_keys=list(in_dict.keys()), out_keys='out_key',
                                                   in_shapes=[value.shape[1:] for value in in_dict.values()],
                                                   hidden_features=[11, 13], non_lins=[nn.ReLU, nn.Identity],
                                                   attention_alpha=0.2, n_heads=[3, 1], attention_dropout=0,
                                                   avg_last_head_attentions=False)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_features == 13
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_features
    assert out_dict[net.out_keys[0]].shape == (100, 5, 13)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-2:]]
    assert net.get_num_of_parameters() == (7 * 11 + 11 * 2) * 3 + (33 * 13 + 13 * 2)

    net.apply(make_module_init_normc(1.0))


def test_graph_attention_block_trainable_self_importance():
    """Test the graph conv block with a trainable self importance scalar"""

    in_dict = build_input_dict(dims=[100, 5, 7])
    adj_matrix_batch = construct_pre_processing_matrix_adj_bar()[0].repeat([100, 1, 1])
    in_dict['adj_matrix'] = adj_matrix_batch
    net: GraphAttentionBlock = GraphAttentionBlock(in_keys=list(in_dict.keys()), out_keys='out_key',
                                                   in_shapes=[value.shape[1:] for value in in_dict.values()],
                                                   hidden_features=[11, 13, 15], non_lins=[nn.ReLU, nn.ReLU, nn.ReLU],
                                                   attention_alpha=0.2, n_heads=[3, 3, 3], attention_dropout=0.5,
                                                   avg_last_head_attentions=True)

    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_features == 15
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_features
    assert tuple(out_dict[net.out_keys[0]].shape) == (100, 5, 15)
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-2:]]
    assert net.get_num_of_parameters() == (7 * 11 + 11 * 2) * 3 + (33 * 13 + 13 * 2) * 3 + ((13 * 3) * 15 + 15 * 2) * 3

    net.apply(make_module_init_normc(1.0))
