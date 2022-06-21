""" Contains a graph neural network block. """
from typing import Any, List, Tuple, Union, Sequence, Dict

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class AggregationLayer(nn.Module):
    """Aggregation layer for message passing withing GNN.

    :param aggregate: The aggregation function to use (max, mean, sum).
    :param pooling_mask: Multiplicative pooling mask for efficient aggregation computation.
    """

    def __init__(self, aggregate: str, pooling_mask: torch.Tensor):
        super(AggregationLayer, self).__init__()

        self.aggregate_str = aggregate
        self.aggregate = self._aggregation_fun(self.aggregate_str)
        self.pooling_mask = pooling_mask

        self.edges = self._edges_from_pooling_mask(torch.transpose(self.pooling_mask, 1, 2))

    @override(nn.Module)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """
        row, col = self.edges.to(t.device)
        aggr = scatter(src=t[:, row], index=col, dim=-2, dim_size=self.pooling_mask.shape[1],
                       reduce=self.aggregate_str)

        # kept for debugging purposes
        # self._assert_scatter_output(t, aggr)

        return aggr

    @classmethod
    def _edges_from_pooling_mask(cls, pooling_mask: torch.Tensor) -> torch.Tensor:
        """Prepare edge list from pooling mask (e.g. for use in scatter reduction).

        :param pooling_mask: The pooling mask to convert into edges.
        :return: Edge matrix corresponding to the pooling mask.
        """
        pm = pooling_mask.squeeze().cpu().numpy()

        edges = []

        for i in range(pm.shape[0]):
            for j in np.nonzero(~np.isnan(pm[i]))[0]:
                edges.append([i, j])

        edges = np.vstack(edges).T
        return torch.from_numpy(edges)

    @classmethod
    def _aggregation_fun(cls, aggregate: str) -> Any:
        """
        :param aggregate: The aggregation function string.
        :return: The actual torch aggregation function.
        """
        if aggregate == "max":
            def maximum(a, dim):
                return torch.max(a, dim=dim)[0]
            return maximum
        elif aggregate == "mean":
            return torch.nanmean
        elif aggregate == "sum":
            return torch.nansum
        else:
            raise ValueError

    def _assert_scatter_output(self, t: torch.Tensor, aggr_scatter: torch.Tensor) -> None:
        """Validation function for efficient scatter implementation.

        :param t: Aggregation source tensor.
        :param aggr_scatter: Aggregation result of scatter.
        """
        aggr_2 = self.aggregate(t.unsqueeze(1) * self.pooling_mask.to(t.device), dim=-2)
        assert torch.allclose(aggr_scatter, aggr_2)


class GNNBlock(ShapeNormalizationBlock):
    """A customizable graph neural network (GNN) block.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param edges: List of graph edges required for message passing (aggregation).
    :param aggregate: The aggregation function to use (max, mean, sum).
    :param hidden_units: List containing the number of hidden units for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    :param with_layer_norm: If True layer normalization is applied.
    :param node2node_aggr: If True node to node message passing is applied.
    :param edge2node_aggr: If True edge to node message passing is applied.
    :param node2edge_aggr: If True node to edge message passing is applied.
    :param edge2edge_aggr: If True edge to edge message passing is applied.
    :param with_node_embedding: If True the node embedding is computed.
    :param with_edge_embedding: If True the edge embedding is computed.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 edges: List[Tuple[int, int]],
                 aggregate: str,
                 hidden_units: List[int],
                 non_lin: Union[str, type(nn.Module)],
                 with_layer_norm: bool,
                 node2node_aggr: bool,
                 edge2node_aggr: bool,
                 node2edge_aggr: bool,
                 edge2edge_aggr: bool,
                 with_node_embedding: bool,
                 with_edge_embedding: bool
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

        self.with_node_embedding = with_node_embedding
        self.with_edge_embedding = with_edge_embedding

        self.hidden_units = hidden_units
        self.n_layers = len(self.hidden_units)
        self.non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.with_layer_norm = with_layer_norm

        # sort edges
        self.edge_list = np.sort(self.edge_list, axis=0)

        # node layers
        if with_node_embedding or node2edge_aggr or edge2node_aggr:
            self.node_layers = nn.ModuleList()
            for i in range(self.n_layers):

                if i == 0:
                    in_dim = self.num_node_features
                else:
                    in_dim = self.hidden_units[i - 1]
                    if self.node2node_aggr:
                        in_dim += self.hidden_units[i - 1]
                    if self.edge2node_aggr:
                        in_dim += self.hidden_units[i - 1]

                if with_node_embedding or (i + 1) < self.n_layers:
                    self.node_layers.append(self._make_sub_layer(in_dim=in_dim, out_dim=self.hidden_units[i]))

        # edge layers
        if with_edge_embedding or edge2node_aggr or node2edge_aggr:
            self.edge_layers = nn.ModuleList()
            for i in range(self.n_layers):

                if i == 0:
                    in_dim = self.num_edge_features
                else:
                    in_dim = self.hidden_units[i - 1]
                    if self.edge2edge_aggr:
                        in_dim += self.hidden_units[i - 1]
                    if self.node2edge_aggr:
                        in_dim += self.hidden_units[i - 1]

                if with_edge_embedding or (i + 1) < self.n_layers:
                    self.edge_layers.append(self._make_sub_layer(in_dim=in_dim, out_dim=self.hidden_units[i]))

        # prepare aggregation layers
        self.aggregate = aggregate
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

        # prepare layer output
        out_dict = dict()

        if self.with_node_embedding:
            out_dict[self.out_keys[0]] = node_embedding

        if self.with_edge_embedding:
            idx = 1 if self.with_node_embedding else 0
            out_dict[self.out_keys[idx]] = edge_embedding

        return out_dict

    def _compute_embeddings(self, n: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the embeddings of node and edge feature matrices.

        :param n: The initial node feature matrix.
        :param e: The initial edge feature matrix.
        :return: Tuple holding the node and edge embedding tensors.
        """

        # iterate layers
        prev_n, prev_e = None, None
        for i in range(self.n_layers):
            first_layer = i == 0
            last_layer = i == self.n_layers - 1

            if self.with_node_embedding or self.node2edge_aggr or self.edge2node_aggr:

                # message passing: node -> node
                if not first_layer:

                    if self.node2node_aggr:
                        n2n = self.n2n_aggregation(prev_n)
                        n = torch.cat((n, n2n), dim=-1)

                    if self.edge2node_aggr:
                        e2n = self.e2n_aggregation(prev_e)
                        n = torch.cat((n, e2n), dim=-1)

                # embed nodes
                if self.with_node_embedding or not last_layer:
                    n = self.node_layers[i](n)

            if self.with_edge_embedding or self.edge2node_aggr or self.node2edge_aggr:

                if not first_layer:

                    if self.edge2edge_aggr:
                        e2e = self.e2e_aggregation(prev_e)
                        e = torch.cat((e, e2e), dim=-1)

                    if self.node2edge_aggr:
                        n2e = self.n2e_aggregation(prev_n)
                        e = torch.cat((e, n2e), dim=-1)

                # embed edges
                if self.with_edge_embedding or not last_layer:
                    e = self.edge_layers[i](e)

            prev_n = n
            prev_e = e

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

    def _make_sub_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """Prepare gnn sublayer stack.

        :param in_dim: Input dimensionality of layer.
        :param out_dim: Output dimensionality of layer.
        :return: Gnn sublayer stack.
        """
        sub_layers = [nn.Linear(in_dim, out_dim)]
        if self.with_layer_norm:
            sub_layers.append(nn.LayerNorm(out_dim))
        sub_layers.append(self.non_lin())
        return nn.Sequential(*sub_layers)

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
        txt += f"\n\thidden_units: {self.hidden_units}, aggr: {self.aggregate}"
        txt += f"\n\tmessage passing:{mp}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
