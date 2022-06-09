"""Contains unit tests for the GNNBlock"""
from typing import Dict

import numpy as np
import torch.nn

from maze.perception.blocks.feed_forward.graph_nn import GNNBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict
from maze.train.utils.train_utils import compute_gradient_norm


def run_gnn_config(node2node_aggr: bool, edge2node_aggr: bool, node2edge_aggr: bool, edge2edge_aggr: bool,
                   with_node_embedding: bool, with_edge_embedding: bool):

    n_nodes = 7
    edges = [(0, 1), (0, 2), (1, 3), (4, 5), (2, 5), (0, 3), (0, 6), (1, 5)]
    n_edges = len(edges)
    n_node_features = 7
    n_edge_features = 9

    for d0, d1 in [([n_nodes, n_node_features], [n_edges, n_edge_features]),
                   ([100, n_nodes, n_node_features], [100, n_edges, n_edge_features])]:
        in_dict = build_multi_input_dict(dims=[d0, d1])

        out_keys = []
        if with_node_embedding:
            out_keys.append("node_embedding")
        if with_edge_embedding:
            out_keys.append("edge_embedding")

        net = GNNBlock(in_keys=["in_key_0", "in_key_1"],
                       out_keys=out_keys,
                       in_shapes=[(n_nodes, n_node_features), [n_edges, n_edge_features]],
                       edges=edges,
                       aggregate="sum",
                       hidden_units=[8, 16, 8],
                       non_lin=torch.nn.ReLU,
                       with_layer_norm=True,
                       node2node_aggr=node2node_aggr,
                       edge2node_aggr=edge2node_aggr,
                       node2edge_aggr=node2edge_aggr,
                       edge2edge_aggr=edge2edge_aggr,
                       with_node_embedding=with_node_embedding,
                       with_edge_embedding=with_edge_embedding)

        net.train()
        optimizer = torch.optim.Adam(params=net.parameters())
        optimizer.zero_grad()

        out_dict = net(in_dict)

        loss = 0.0
        if net.with_node_embedding:
            loss += out_dict["node_embedding"].mean()
        if net.with_edge_embedding:
            loss += out_dict["edge_embedding"].mean()

        # propagate gradients
        loss.backward()
        optimizer.step()

        # compute gradient and l2 norm
        compute_gradient_norm(net.parameters())
        sum([param.norm() for param in net.parameters()])

        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        if net.with_node_embedding:
            assert np.all(~np.isnan(out_dict["node_embedding"].detach().cpu().numpy()))
            assert out_dict[net.out_keys[0]].shape[-1] == net.hidden_units[-1]
        if net.with_edge_embedding:
            assert np.all(~np.isnan(out_dict["edge_embedding"].detach().cpu().numpy()))
            idx = 1 if net.with_node_embedding else 0
            assert out_dict[net.out_keys[idx]].shape[-1] == net.hidden_units[-1]

        assert set(net.out_keys).issubset(set(out_dict.keys()))


def test_gnn_block():
    """ perception test """

    # node and edge embedding
    run_gnn_config(node2node_aggr=True, edge2node_aggr=True, node2edge_aggr=True, edge2edge_aggr=True,
                   with_node_embedding=True, with_edge_embedding=True)
    run_gnn_config(node2node_aggr=True, edge2node_aggr=False, node2edge_aggr=False, edge2edge_aggr=True,
                   with_node_embedding=True, with_edge_embedding=True)

    # node only embedding
    run_gnn_config(node2node_aggr=True, edge2node_aggr=False, node2edge_aggr=False, edge2edge_aggr=False,
                   with_node_embedding=True, with_edge_embedding=False)
    run_gnn_config(node2node_aggr=True, edge2node_aggr=True, node2edge_aggr=True, edge2edge_aggr=True,
                   with_node_embedding=True, with_edge_embedding=False)

    # edge only embedding
    run_gnn_config(node2node_aggr=False, edge2node_aggr=False, node2edge_aggr=False, edge2edge_aggr=True,
                   with_node_embedding=False, with_edge_embedding=True)
    run_gnn_config(node2node_aggr=True, edge2node_aggr=True, node2edge_aggr=True, edge2edge_aggr=True,
                   with_node_embedding=False, with_edge_embedding=True)
