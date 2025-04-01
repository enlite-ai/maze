"""Contains a gnn block that uses Pytorch Geometric to support different types of GNNs"""
from collections import OrderedDict
from typing import Sequence, Callable

import torch
from torch import nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GraphConv

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


SUPPORTED_GNNS = ['gcn', 'sage', 'graph_conv']

class GNNLayerPyG(nn.Module):
    """Simple graph convolution layer.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param bias: Whether to include bias in the PyG layer.
    :param gnn_type: The type of GNN layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, gnn_type: str) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if gnn_type.lower() == 'gcn':
            self.gnn_layer = GCNConv(in_features, out_features, bias=bias)
        elif gnn_type.lower() == 'sage':
            self.gnn_layer = SAGEConv(in_features, out_features, bias=bias)
        elif gnn_type.lower() == 'graph_conv':
            self.gnn_layer = GraphConv(in_features, out_features, bias=bias)
        else:
            raise ValueError(f'Unsupported GNN type: {gnn_type}. Supported GNNs are {SUPPORTED_GNNS}')

        self.gnn_type = gnn_type

    @override(nn.Module)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.

        :param x: Node feature matrix, [n_nodes, in_features] or [B, n_nodes, in_features].
        :param edge_index: The graph connectivity in COO format, [2, E] or [B, 2, E].
        :param edge_attr: The edge attributes, [E, D] or [B, E, D]. D is the edge attribute dimension.
        :return: Output tensor.
        """
        if x.dim() == 2:  # Single graph (no batch)
            if self.gnn_type in ['gcn', 'graph_conv']:
                return self.gnn_layer(x, edge_index, edge_weight=edge_attr if edge_attr.dim() == 1 else None)
            else:
                # SAGEConv does not use edge_attr
                return self.gnn_layer(x, edge_index)

        elif x.dim() == 3:  # Batched graphs
            batch_size, num_nodes, in_features = x.shape

            # Flatten batch for efficient processing
            x_flat = x.view(-1, in_features)  # [B*N, in_features]
            edge_index_flat = torch.cat([edge_index[b] + b * num_nodes for b in range(batch_size)], dim=1)
            edge_attr_flat = torch.cat([edge_attr[b] for b in range(batch_size)],
                                       dim=0) if edge_attr is not None else None

            if self.gnn_type in ['gcn', 'graph_conv']:
                out = self.gnn_layer(
                    x_flat, edge_index_flat,
                    edge_weight=edge_attr_flat if (edge_attr_flat is not None and edge_attr_flat.dim() == 1) else None
                )
            else:
                out = self.gnn_layer(x_flat, edge_index_flat)

            return out.view(batch_size, num_nodes, -1)  # Reshape back

        else:
            raise ValueError(f"Unexpected x shape: {x.shape}, expected 2D or 3D.")

    @override(nn.Module)
    def __repr__(self):
        txt = f'{self.gnn_type}: ({self.in_features} -> {self.out_features})'
        txt += ' (with bias)' if self.bias else ' (without bias)'
        return txt


class GNNBlockPyG(ShapeNormalizationBlock):
    """A block containing subsequent GNN layers.

    :param in_keys: Two keys identifying the node feature matrix and the graph connectivity list of edges.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_features: List containing the number of hidden features for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    :param bias: Whether to include bias in the GNN layers.
    :param gnn_type: The type of GNN layer.
    """

    def __init__(
        self, in_keys: str | list[str], out_keys: str | list[str],
        in_shapes: Sequence[int] | list[Sequence[int]],
        hidden_features: list[int],
        non_lin: str | nn.Module,
        bias: bool,
        gnn_type: str,
    ):

        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=[3]*3, out_num_dims=3)

        self.gnn_type = gnn_type
        self.bias = bias

        assert len(self.in_keys) == 3, \
            'There should be three input keys: node feature matrix, graph edge_index, and edge attributes.'

        # Specify dummy dict creation function for edge_index:
        self.dummy_dict_creators[1] = _dummy_edge_index_factory(self.in_shapes[1], self.in_shapes[0][0])

        # Init class objects
        self.input_features = self.in_shapes[0][-1]
        self.hidden_features = hidden_features
        self.output_features = self.hidden_features[-1]

        self.non_lin: type[nn.Module] = Factory(base_type=nn.Module).type_from_name(non_lin)

        # Form layers dictionary
        layer_dict = self.build_layer_dict()

        # Compile network
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
            f"Feature dimension should fit: {node_feat.shape[-1]} vs {self.input_features}"
        assert edge_index.shape[-1] == edge_attr.shape[-2], \
            "Number of edges must be consistent"

        # forward pass
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

        This could be overwritten by derived layers (e.g. to get a 'BatchNormalizedConvolutionBlock').

        :return: Ordered dictionary of torch modules [str, nn.Module].
        """
        layer_dict = OrderedDict()
        in_feats = self.input_features

        for layer_idx, out_feats in enumerate(self.hidden_features):
            layer_dict[f'{self.gnn_type}_{layer_idx}'] = GNNLayerPyG(
                in_features=in_feats,
                out_features=out_feats,
                bias=self.bias,
                gnn_type=self.gnn_type,
            )
            # Add activation function only for intermediate layers
            if layer_idx < len(self.hidden_features) - 1:
                layer_dict[f'activation_{layer_idx}_{self.non_lin.__name__}'] = self.non_lin()
            in_feats = out_feats

        return layer_dict

    def __repr__(self):
        txt = f'{self.__class__.__name__}'
        txt += f'({self.non_lin.__name__})'
        txt += '\n\t' + f'({self.input_features}->' + '->'.join([f'{h}' for h in self.hidden_features]) + ')'
        txt += f'\n\tBias: {self.bias}'
        txt += f'\n\tOut Shapes: {self.out_shapes()}'
        return txt