""" Contains a graph neural network block. """
from typing import Any, List, Tuple, Union, Sequence, Dict

import numpy as np
import torch
from torch import nn

from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class AggregationLayer(nn.Module):
    """Aggregation layer for message passing withing GNN.

    :param aggregate: The aggregation function to use (max, mean, sum).
    :param pooling_mask: Multiplicative pooling mask for efficient aggregation computation.
    """

    def __init__(self, aggregate: str, pooling_mask: torch.Tensor):
        super(AggregationLayer, self).__init__()

        self.aggregate = self._aggregation_fun(aggregate)
        self.pooling_mask = pooling_mask

    @override(nn.Module)
    def forward(self, n: torch.Tensor) -> torch.Tensor:
        n = n.unsqueeze(1)
        return self.aggregate(n * self.pooling_mask, dim=-2)

    @classmethod
    def _aggregation_fun(cls, aggregate: str) -> Any:
        """
        :param aggregate: The aggregation function string.
        :return: The actual torch aggregation function.
        """
        if aggregate == "max":
            return torch.max
        elif aggregate == "mean":
            return torch.nanmean
        elif aggregate == "sum":
            return torch.nansum
        else:
            raise ValueError


class GNNBlock(ShapeNormalizationBlock):
    """A customizable graph neural network (GNN) block.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param edges: List of graph edges required for message passing (aggregation).
    :param aggregate: The aggregation function to use (max, mean, sum).
    :param embed_dim: Dimensionality of node and edge embedding space.
    :param n_layers: Number of hidden layers.
    :param node2node_aggr: If True node to node message passing is applied.
    :param edge2node_aggr: If True edge to node message passing is applied.
    :param node2edge_aggr: If True node to edge message passing is applied.
    :param edge2edge_aggr: If True edge to edge message passing is applied.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 edges: List[Tuple[int, int]],
                 aggregate: str,
                 embed_dim: int,
                 n_layers: int,
                 node2node_aggr: bool,
                 edge2node_aggr: bool,
                 node2edge_aggr: bool,
                 edge2edge_aggr: bool
                 ):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes,
                         in_num_dims=[3, 3], out_num_dims=[3, 3])

        edge_list = np.vstack(edges).T

        self.num_nodes = in_shapes[0][0]
        self.num_node_features = in_shapes[0][1]
        self.num_edge_features = in_shapes[1][1]

        self.num_edges = edge_list.shape[1]
        self.edge_list = edge_list

        self.node2node_aggr = node2node_aggr
        self.edge2node_aggr = edge2node_aggr
        self.node2edge_aggr = node2edge_aggr
        self.edge2edge_aggr = edge2edge_aggr

        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # sort edges
        self.edge_list = np.sort(self.edge_list, axis=0)

        # node layers
        self.node_layers = []
        for i in range(self.n_layers):

            if i == 0:
                in_dim = self.num_node_features
            else:
                in_dim = 2 * self.embed_dim if self.edge2node_aggr else self.embed_dim

            self.node_layers.append(nn.Sequential(nn.Linear(in_dim, self.embed_dim),
                                                  nn.ReLU(), nn.LayerNorm(self.embed_dim)))

        # edge layers
        self.edge_layers = []
        for i in range(self.n_layers):

            if i == 0:
                in_dim = self.num_edge_features
            else:
                in_dim = 2 * self.embed_dim if self.node2edge_aggr else self.embed_dim

            self.edge_layers.append(nn.Sequential(nn.Linear(in_dim, self.embed_dim),
                                                  nn.ReLU(), nn.LayerNorm(self.embed_dim)))

        # prepare aggregation layers
        pooling_matrices = self._prepare_pooling_matrices()
        self.n2n_aggregation = AggregationLayer(aggregate=aggregate, pooling_mask=pooling_matrices[0])
        self.e2n_aggregation = AggregationLayer(aggregate=aggregate, pooling_mask=pooling_matrices[1])
        self.n2e_aggregation = AggregationLayer(aggregate=aggregate, pooling_mask=pooling_matrices[2])
        self.e2e_aggregation = AggregationLayer(aggregate=aggregate, pooling_mask=pooling_matrices[3])

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # check input tensor
        node_tensor = block_input[self.in_keys[0]]
        edge_tensor = block_input[self.in_keys[1]]
        assert node_tensor.ndim == self.in_num_dims[0]
        assert edge_tensor.ndim == self.in_num_dims[1]

        # forward pass
        node_embedding, edge_embedding = self._compute_embeddings(node_tensor, edge_tensor)

        # check output tensor
        assert node_embedding.ndim == self.out_num_dims[0]
        assert edge_embedding.ndim == self.out_num_dims[1]

        return {self.out_keys[0]: node_embedding, self.out_keys[1]: edge_embedding}

    def _compute_embeddings(self, n: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the embeddings of node and edge feature matrices.

        :param n: The initial node feature matrix.
        :param e: The initial edge feature matrix.
        :return: Tuple holding the node and edge embedding tensors.
        """

        # iterate layers
        for i in range(self.n_layers):
            last_layer = i == self.n_layers - 1

            # embed nodes
            n = self.node_layers[i](n)

            # embed edges
            e = self.edge_layers[i](e)

            # message passing: node -> node
            if self.node2node_aggr:
                n = self.n2n_aggregation(n)

            # message passing: edge -> edge
            if self.edge2edge_aggr:
                e = self.e2e_aggregation(e)

            # message passing: edge -> node I
            e2n = None
            if self.edge2node_aggr and not last_layer:
                e2n = self.e2n_aggregation(e)

            # message passing: node -> edge I
            n2e = None
            if self.node2edge_aggr and not last_layer:
                n2e = self.n2e_aggregation(n)

            # message passing: edge -> node II
            if self.edge2node_aggr and not last_layer:
                assert n.shape == e2n.shape
                n = torch.concat((n, e2n), dim=-1)

            # message passing: node -> edge II
            if self.node2edge_aggr and not last_layer:
                assert e.shape == n2e.shape
                e = torch.concat((e, n2e), dim=-1)

        return n, e

    def _prepare_pooling_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares multiplicative pooling matrices for efficient feature aggregation.

        :return: Tuple holding the four pooling matrices.
        """

        # prepare node -> node pooling mask
        n2n_pooling_mask = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        n2n_pooling_mask[:] = np.nan

        for i in range(self.num_nodes):
            idxs = self.edge_list[1, self.edge_list[0, :] == i]
            n2n_pooling_mask[i, idxs] = 1.0
            n2n_pooling_mask[idxs, i] = 1.0

        n2n_pooling_mask[range(self.num_nodes), range(self.num_nodes)] = 1.0

        n2n_pooling_mask = torch.from_numpy(n2n_pooling_mask).unsqueeze(-1).unsqueeze(0)

        # prepare edge -> node pooling mask
        e2n_pooling_mask = np.zeros((self.num_nodes, self.num_edges), dtype=np.float32)
        e2n_pooling_mask[:] = np.nan

        for i in range(self.num_edges):
            e2n_pooling_mask[self.edge_list[:, i], i] = 1.0

        e2n_pooling_mask = torch.from_numpy(e2n_pooling_mask).unsqueeze(-1).unsqueeze(0)

        # prepare node -> edge pooling mask
        n2e_pooling_mask = np.zeros((self.num_edges, self.num_nodes), dtype=np.float32)
        n2e_pooling_mask[:] = np.nan

        for i in range(self.num_edges):
            n2e_pooling_mask[i, self.edge_list[:, i]] = 1.0

        n2e_pooling_mask = torch.from_numpy(n2e_pooling_mask).unsqueeze(-1).unsqueeze(0)

        # prepare edge -> edge pooling mask
        e2e_pooling_mask = np.zeros((self.num_edges, self.num_edges), dtype=np.float32)
        e2e_pooling_mask[:] = np.nan

        for i in range(self.num_edges):
            e = self.edge_list[:, i]
            idxs = (self.edge_list[0, :] == e[0]) | (self.edge_list[0, :] == e[1]) \
                   | (self.edge_list[1, :] == e[0]) | (self.edge_list[1, :] == e[1])
            e2e_pooling_mask[i, idxs] = 1.0
            e2e_pooling_mask[idxs, i] = 1.0

        e2e_pooling_mask[range(self.num_edges), range(self.num_edges)] = 1.0

        e2e_pooling_mask = torch.from_numpy(e2e_pooling_mask).unsqueeze(-1).unsqueeze(0)

        return n2n_pooling_mask, e2n_pooling_mask, n2e_pooling_mask, e2e_pooling_mask

    def __repr__(self):

        mp = ""
        if self.node2node_aggr:
            mp += " n2n"
        if self.edge2edge_aggr:
            mp += " e2e"
        if self.node2edge_aggr:
            mp += " n2e"
        if self.edge2node_aggr:
            mp += " e2n"

        txt = f"{GNNBlock.__name__}"
        txt += f"\n\tIn Shapes: {self.in_shapes}"
        txt += f"\n\tn_layers: {self.n_layers}, embed_dim: {self.embed_dim}"
        txt += f"\n\tmessage passing:{mp}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
