""" Contains a gnn block that uses Pytorch Geometric to support different types of GNNs"""
from collections import OrderedDict
from typing import Sequence, Callable, Any

import torch
from torch import nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


def _dummy_edge_index_factory(shape: Sequence[int], n_nodes: int) -> Callable[[], torch.Tensor]:
    """
    Factory for creating a dummy edge index tensor.

    :param shape: The shape of the edge tensor.
    :param n_nodes: The number of nodes in the edge tensor
    :return: A function that creates a dummy edge index tensor
    """
    def create_dummy_edge_index_tensor() -> torch.Tensor:
        """
        Create a dummy edge index tensor
        :return: Dummy edge index tensor
        """
        return torch.randint(0, n_nodes, tuple(shape)).unsqueeze(dim=0)

    return create_dummy_edge_index_tensor


SUPPORTED_GNNS = ['gcn', 'sage', 'graph_conv', 'gat']

class GNNLayerPyG(nn.Module):
    """Simple graph neural network layer.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param gnn_type: The type of GNN layer.
    :param gnn_kwargs:   Additional keyword arguments passed to the underlying PyG layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gnn_type: str,
        gnn_kwargs: dict[str, Any] | None
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gnn_type = gnn_type.lower()
        self.gnn_kwargs = gnn_kwargs if gnn_kwargs is not None else {}

        if self.gnn_type == 'gcn':
            self.gnn_layer = GCNConv(
                in_channels=in_features,
                out_channels=out_features,
                **self.gnn_kwargs
            )
        elif self.gnn_type == 'sage':
            self.gnn_layer = SAGEConv(
                in_channels=in_features,
                out_channels=out_features,
                **self.gnn_kwargs
            )
        elif self.gnn_type == 'graph_conv':
            self.gnn_layer = GraphConv(
                in_channels=in_features,
                out_channels=out_features,
                **self.gnn_kwargs
            )
        elif self.gnn_type == 'gat':
            # For GAT with edge attributes, set "edge_dim" in gnn_kwargs.
            self.gnn_layer = GATConv(
                in_channels=in_features,
                out_channels=out_features,
                **self.gnn_kwargs
            )
        else:
            raise ValueError(f'Unsupported GNN type: {gnn_type}. Supported GNNs are {SUPPORTED_GNNS}')

    @override(nn.Module)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.

        :param x: Node feature matrix, [n_nodes, in_features] or [B, n_nodes, in_features].
        :param edge_index: The graph connectivity in COO format, [2, E] or [B, 2, E].
        :param edge_attr: The edge attributes, [E, D] or [B, E, D]. D is the edge attribute dimension.
        :return: Output tensor.
        """

        reshaped = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            edge_attr = edge_attr.unsqueeze(0)
            reshaped = True

        # Expect x with shape [B, n_nodes, in_features].
        batch_size, n_nodes, in_features = x.shape

        # Flatten the batch for node features: (B*n_nodes, in_features)
        x_flat = x.view(-1, in_features)

        # Flatten the batch for edges
        edge_index_list = []
        edge_attr_list = []
        for b in range(batch_size):
            offset = b * n_nodes
            edge_index_batch = edge_index[b] + offset  # (2, E)
            edge_index_list.append(edge_index_batch)
            if edge_attr is not None:
                edge_attr_list.append(edge_attr[b])  # (E, D) or (E,)

        edge_index_flat = torch.cat(edge_index_list, dim=1)  # => (2, sum(E_b))
        edge_attr_flat = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None

        if self.gnn_type in ['gcn', 'graph_conv']:
            # Interpret 1D edge_attr as edge_weight if provided
            edge_weight = None
            if edge_attr_flat is not None and edge_attr_flat.dim() == 1:
                edge_weight = edge_attr_flat

            out_flat = self.gnn_layer(x_flat, edge_index_flat, edge_weight=edge_weight)

        elif self.gnn_type == 'sage':
            out_flat = self.gnn_layer(x_flat, edge_index_flat)

        elif self.gnn_type == 'gat':
            # GATConv can use edge_attr if "edge_dim" is provided in gnn_kwargs
            if self.gnn_layer.edge_dim is not None:
                assert edge_attr_flat.shape[-1] == self.gnn_layer.edge_dim, \
                    f'The edge feature size: {edge_attr_flat.shape[-1]} must match the edge_dim: {self.gnn_layer.edge_dim}'
            out_flat = self.gnn_layer(x_flat, edge_index_flat, edge_attr=edge_attr_flat)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

        # Reshape back to (B, n_nodes, out_features)
        out = out_flat.view(batch_size, n_nodes, -1)

        # Reshape back if we had inserted a batch dimension of size 1
        if reshaped:
            out = out.squeeze(0)  # => (n_nodes, out_features)

        return out

    @override(nn.Module)
    def __repr__(self):
        txt = f'{self.gnn_type}: ({self.in_features} -> {self.out_features})'
        txt += f', kwargs={self.gnn_kwargs}'
        return txt


class GNNBlockPyG(ShapeNormalizationBlock):
    """A block containing subsequent GNN layers.

    :param in_keys: Two keys identifying the node feature matrix and the graph connectivity list of edges.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_features: List containing the number of hidden features for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    :param gnn_type: The type of GNN layer
    :param gnn_kwargs: Extra kwargs to pass to the GNN layers.
    """

    def __init__(
        self, in_keys: str | list[str], out_keys: str | list[str],
        in_shapes: Sequence[int] | list[Sequence[int]],
        hidden_features: list[int],
        non_lin: str | nn.Module,
        gnn_type: str,
        gnn_kwargs: dict[str, Any] | None
    ):

        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=[3]*3, out_num_dims=3)

        self.gnn_type = gnn_type
        self.gnn_kwargs = gnn_kwargs if gnn_kwargs is not None else {}

        assert len(self.in_keys) == 3, \
            f"Expected three input keys, got {len(self.in_keys)}: {self.in_keys}"

        # Specify dummy dict creation function for edge_index:
        self.dummy_dict_creators[1] = _dummy_edge_index_factory(self.in_shapes[1], self.in_shapes[0][0])

        self.input_features = self.in_shapes[0][-1]

        if (gnn_type == 'gat' and gnn_kwargs is not None and 'heads' in gnn_kwargs
                and ('concat' not in gnn_kwargs or gnn_kwargs['concat'])):
            self.output_features = hidden_features[-1] * gnn_kwargs['heads']
        else:
            self.output_features = hidden_features[-1]

        self.hidden_features = hidden_features

        self.non_lin: type[nn.Module] = Factory(base_type=nn.Module).type_from_name(non_lin)

        # Create the GNN layers
        layer_dict = self.build_layer_dict()
        self.net = nn.Sequential(layer_dict)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface."""
        # Check input tensors
        node_feat = block_input[self.in_keys[0]]
        edge_index = block_input[self.in_keys[1]]
        edge_attr = block_input[self.in_keys[2]]

        assert node_feat.ndim == self.in_num_dims[0]
        assert edge_index.ndim == self.in_num_dims[1]
        assert edge_attr.ndim == self.in_num_dims[2]

        assert node_feat.shape[-1] == self.input_features, \
            f"Mismatch in node feature dimension: {node_feat.shape[-1]} vs expected {self.input_features}"
        assert edge_index.shape[-1] == edge_attr.shape[-2], \
            (f"Number of edges (E) must be consistent between edge_index: {edge_index.shape[-1]} "
             f"and edge_attr: {edge_attr.shape[-2]}")

        # Forward pass
        x = node_feat
        for layer in self.net:
            if isinstance(layer, GNNLayerPyG):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)

        # check output tensor
        assert x.ndim == self.out_num_dims[0], \
            f"The output number of dimensions {x.ndim} must be {self.out_num_dims[0]}"
        assert x.shape[-1] == self.output_features, \
            f"The output feature dimension size {x.shape[-1]} must be {self.output_features}"

        return {self.out_keys[0]: x}

    def build_layer_dict(self) -> OrderedDict:
        """Compiles a block-specific dictionary of network layers.
        :return: Ordered dictionary of torch modules
        """
        layer_dict = OrderedDict()
        in_feats = self.input_features

        for layer_idx, out_feats in enumerate(self.hidden_features):

            layer_dict[f'{self.gnn_type}_{layer_idx}'] = GNNLayerPyG(
                in_features=in_feats,
                out_features=out_feats,
                gnn_type=self.gnn_type,
                gnn_kwargs=self.gnn_kwargs
            )
            # Insert activation function after each hidden layer except the last
            if layer_idx < len(self.hidden_features) - 1:
                layer_name = f'activation_{layer_idx}_{self.non_lin.__name__}'
                layer_dict[layer_name] = self.non_lin()

            if self.gnn_type == 'gat' and 'heads' in self.gnn_kwargs and \
                    ('concat' not in self.gnn_kwargs or self.gnn_kwargs['concat']):
                in_feats = out_feats * self.gnn_kwargs['heads']
            else:
                in_feats = out_feats

        return layer_dict

    def __repr__(self):
        txt = (
            f"{self.__class__.__name__}({self.non_lin.__name__})\n"
            f"\t({self.input_features}->" + "->".join([f"{h}" for h in self.hidden_features]) + ")\n"
        )
        txt += f"\n\tGNN kwargs: {self.gnn_kwargs}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
