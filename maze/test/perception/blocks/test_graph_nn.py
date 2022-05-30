"""Contains unit tests for the GNNBlock"""
from typing import Dict

import torch.nn

from maze.perception.blocks.feed_forward.graph_nn import GNNBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


def test_GNN_block():
    """ perception test """

    n_nodes = 7
    edges = [(0, 1), (0, 2), (1, 3), (4, 5), (2, 5), (0, 3), (0, 6), (1, 5)]
    n_edges = len(edges)
    n_node_features = 7
    n_edge_features = 9

    for d0, d1 in [([n_nodes, n_node_features], [n_edges, n_edge_features]),
                   ([100, n_nodes, n_node_features], [100, n_edges, n_edge_features])]:
        in_dict = build_multi_input_dict(dims=[d0, d1])

        net = GNNBlock(in_keys=["in_key_0", "in_key_1"],
                       out_keys=["node_embedding", "edge_embedding"],
                       in_shapes=[(n_nodes, n_node_features), [n_edges, n_edge_features]],
                       edges=edges,
                       aggregate="sum",
                       embed_dim=16,
                       n_layers=3,
                       non_lin=torch.nn.ReLU,
                       with_layer_norm=True,
                       node2node_aggr=True,
                       edge2node_aggr=True,
                       node2edge_aggr=True,
                       edge2edge_aggr=True)

        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        assert set(net.out_keys).issubset(set(out_dict.keys()))
        assert out_dict[net.out_keys[0]].shape[-1] == net.embed_dim
        assert out_dict[net.out_keys[1]].shape[-1] == net.embed_dim
