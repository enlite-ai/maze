""" Contains Graph-convolutional perception blocks layers.
    Credit is given to https://arxiv.org/abs/1609.02907 and
    https://github.com/meliketoy/graph-cnn.pytorch/blob/master/layers.py
 """
from collections import OrderedDict
from typing import Union, List, Sequence, Dict, Optional, Callable

import numpy as np
import torch
from torch import nn as nn
from torch.nn.parameter import Parameter

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class GraphAdjacencyMethods:
    """Base class for all graph blocks, implementing all methods needed for processing the adjacency matrix"""

    @classmethod
    def _dummy_symmetric_adj_tensor_factory(cls, in_shape: Sequence[int]) -> Callable[[], torch.Tensor]:
        """Factory for creating a dummy tensor that is also a binary symmetric adjacency matrix.

        :param in_shape: The in_shape we want to create a tensor for.
        :return: A functional taking no arguments returning a tensor.
        """

        def create_binary_sym_tensor() -> torch.Tensor:
            """Create a binary symmetric test matrix for shape inference.

            :return: A binary symmetric matrix with one additional batch dimension.
            """
            xx_np = np.random.randint(0, 2, size=in_shape).astype(np.float32)
            xx = torch.from_numpy(xx_np)
            xx_sym = (xx + torch.transpose(xx, dim0=-2, dim1=-1))
            xx_sym[xx_sym > 0] = 1
            xx_sym = xx_sym.unsqueeze(dim=0)
            return xx_sym

        return create_binary_sym_tensor

    @classmethod
    def preprocess_adj_to_adj_bar(cls, adj: torch.Tensor) \
            -> Union[torch.Tensor, np.ndarray]:
        """Transform the adjacency matrix needed for the computation.

        Since repeated application of forward computation on the standard adjacency matrix can lead to numerical
        instabilities and exploding/vanishing gradients when used in a deep neural network model. To alleviate this
        problem, the following renormalization trick is used:

        A^bar := A + I_n

        :param adj: The standard adjacency matrix.
        :return: The processed adjacency matrix A^hat.
        """

        batch_dim = 0
        if len(adj.shape) > 2:
            # If a batch of adjacency matrices is given compute only once and repeat
            batch_dim = adj.shape[0]
        else:
            adj = adj.unsqueeze(0)
        # Our framework currently does not naturally support edge features  and  is  limited  to  undirected  graphs  (
        # weighted  or  unweighted). Results  on  NELL  however show that it is possible to handle both directed edges
        # and edge features by representing the original directed graph as an undirected bipartite graph with additional
        # nodes that represent edges in the original graph (see Section 5.1 for details).
        # - https://arxiv.org/abs/1609.02907 THUS --> adjacency matrix has to be symmetric (-> thus square)
        assert adj.shape[-1] == adj.shape[-2], 'The adj matrix should be a square matrix'
        assert torch.allclose(adj, torch.transpose(adj, dim0=-2, dim1=-1)), 'The adj matrix should be symmetric'

        # Add the feature of the node itself. For example, the first row of the result matrix should contain features of
        # node A too.
        adj_bar = adj + torch.eye(adj.shape[-2], device=adj.device).repeat(adj.shape[0], 1, 1)

        if batch_dim == 0:
            adj_bar = adj_bar.squeeze(0)

        return adj_bar

    @classmethod
    def preprocess_adj_to_adj_hat(cls, adj: torch.Tensor,
                                  self_importance_scalar: Optional[torch.Tensor] = torch.tensor(1)) \
            -> Union[torch.Tensor, np.ndarray]:
        """Transform the adjacency matrix needed for the computation.

        Since repeated application of forward computation on the standard adjacency matrix can lead to numerical
        instabilities and exploding/vanishing gradients when used in a deep neural network model. To alleviate this
        problem, the following renormalization trick is used:

        A^bar := A + I_n * :param self_importance_scalar
        D^bar_ii := sum_j A^bar_ij
        then: A^hat := D^bar^(-1/2) A^bar D^bar^(-1/2)

        :param adj: The standard adjacency matrix.
        :param self_importance_scalar: Determine how important a node is to itself relative to other nodes.
            (default 1, meaning it is just as important as other nodes) - could be treated as a trainable parameter.
        :return: The normalized adjacency matrix A^hat.
        """

        batch_dim = 0
        if len(adj.shape) > 2:
            # If a batch of adjacency matrices is given compute only once and repeat
            batch_dim = adj.shape[0]
        else:
            adj = adj.unsqueeze(0)
        # Our framework currently does not naturally support edge features  and  is  limited  to  undirected  graphs  (
        # weighted  or  unweighted). Results  on  NELL  however show that it is possible to handle both directed edges
        # and edge features by representing the original directed graph as an undirected bipartite graph with additional
        # nodes that represent edges in the original graph (see Section 5.1 for details).
        # - https://arxiv.org/abs/1609.02907 THUS --> adjacency matrix has to be symmetric (-> thus square)
        assert adj.shape[-1] == adj.shape[-2], 'The adj matrix should be a square matrix'
        assert torch.allclose(adj, torch.transpose(adj, dim0=-2, dim1=-1)), 'The adj matrix should be symmetric'

        # Add the feature of the node itself. For example, the first row of the result matrix should contain features of
        # node A too.
        adj_bar = adj + self_importance_scalar * torch.eye(adj.shape[-2], device=adj.device).repeat(adj.shape[0], 1, 1)

        # Row-normalize sparse matrix, so that the adjacency matrix is scaled by both rows and columns.
        rowsum = adj_bar.sum(-1)
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.

        r_mat_inv_sqrt = torch.diag_embed(r_inv_sqrt)
        row_norm = torch.matmul(adj_bar, r_mat_inv_sqrt)
        adj_hat = torch.matmul(torch.transpose(row_norm, dim0=-2, dim1=-1), r_mat_inv_sqrt)

        if batch_dim == 0:
            adj_hat = adj_hat.squeeze(0)

        return adj_hat


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param bias: Specify if a bias should be used.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(size=(in_features, out_features), dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.empty(size=(out_features,), dtype=torch.float32))
        else:
            self.bias = None

    @override(nn.Module)
    def forward(self, xx: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the layer with the modified adjacency matrix a_hat.

        :param xx: Input feature tensor.
        :param adj_hat: Preprocessed (normalized in row and column) adjacency matrix.
        :return: Output tensor.
        """
        support = torch.matmul(xx, self.weight)
        output = torch.matmul(adj_hat, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    @override(nn.Module)
    def __repr__(self):
        txt = self.__class__.__name__ + ': (' \
              + str(self.in_features) + ' -> ' \
              + str(self.out_features) + ')'
        if self.bias is not None:
            txt += ' (with bias)'
        else:
            txt += ' (without bias)'
        return txt


class GraphConvBlock(ShapeNormalizationBlock, GraphAdjacencyMethods):
    """A block containing multiple subsequent graph convolution stacks.

    One convolution stack consists of one graph convolution in addition to an activation layer. The block expects the
    input tensors to have the form:
    - Feature matrix: first in_key: (batch-dim, num-of-nodes, feature-dim)
    - Adjacency matrix: second in_key: (batch-dim, num-of-nodes, num-of-nodes) (also symmetric)
    And returns a tensor of the form (batch-dim, num-of-nodes, feature-out-dim).

    :param in_keys: Two keys identifying the feature matrix and adjacency matrix respectively.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_features: List containing the number of hidden features for hidden layers.
    :param bias: Specify if a bias should be used at each layer (can be list or single).
    :param non_lins: The non-linearity/ies to apply after each layer (the same in all layers, or a list corresponding
        to each layer).
    :param node_self_importance: Specify how important a given node is to itself (default should be 1).
    :param trainable_node_self_importance: Specify if the node_self_importance should be a constant or a trainable
        parameter with init value :param node_self_importance.
    :param preprocess_adj: Specify whether to preprocess the adjacency, that is compute:
        adj^ := D^bar^(-1/2) A^bar D^bar^(-1/2) in every forward pass for the whole bach. If this is set to false, the
        already normalized adj^ is expected as an input. Here A^bar := A + I_n * :param self_importance_scalar, and
        D^bar_ii := sum_j A^bar_ij.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], hidden_features: List[int],
                 bias: Union[bool, List[bool]], non_lins: Union[str, type(nn.Module), List[str], List[type(nn.Module)]],
                 node_self_importance: float,
                 trainable_node_self_importance: bool, preprocess_adj: bool):

        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=[3, 3],
                         out_num_dims=3)

        # Assertions
        assert len(self.in_keys) == 2, 'There should be two input keys, feature matrix + adjacency matrix'
        assert self.in_shapes[0][-2] == self.in_shapes[1][-1], 'The node dimension of the feature matrix should be ' \
                                                               'the same as the adjacency matrix\'s rows and ' \
                                                               f'columns {self.in_shapes}'
        assert self.in_shapes[1][-1] == self.in_shapes[1][-2], 'The adjacency matrix has to be a square matrix'

        # Specify dummy dict creation function for adjacency matrix:
        self.dummy_dict_creators[1] = self._dummy_symmetric_adj_tensor_factory(self.in_shapes[1])

        # Init class objects
        self.input_features = self.in_shapes[0][-1]
        self.hidden_features = hidden_features
        self.output_features = self.hidden_features[-1]

        self.preprocess_adj = preprocess_adj

        # Create list of non-linearity's for each layer
        non_lins = non_lins if isinstance(non_lins, list) else [non_lins] * len(self.hidden_features)
        self.non_lins: List[type(nn.Module)] = [Factory(base_type=nn.Module).type_from_name(non_lin)
                                                for non_lin in non_lins]
        # Create list of biases for each layer
        self.bias: List[bool] = bias if isinstance(bias, list) else [bias] * len(self.hidden_features)

        # Initialize node-self-importance scalar
        self.node_self_importance = torch.tensor(node_self_importance, dtype=torch.float32, requires_grad=False)

        # Initialize node-self-importance scalar as trainable parameter if applicable
        if trainable_node_self_importance:
            # Save the default value for weight init
            self.node_self_importance_default = self.node_self_importance
            self.node_self_importance = Parameter(self.node_self_importance)
            assert self.node_self_importance.requires_grad is True
            assert self.node_self_importance_default.requires_grad is False

        # compile layer dictionary
        layer_dict = self.build_layer_dict()

        # compile network
        self.net = nn.Sequential(layer_dict)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """
        # check input tensor
        feat_tensor = block_input[self.in_keys[0]]
        adj_tensor = block_input[self.in_keys[1]]

        assert feat_tensor.ndim == self.in_num_dims[0]
        assert adj_tensor.ndim == self.in_num_dims[1]

        assert feat_tensor.shape[-1] == self.input_features, 'Feature dimension should fit'
        assert feat_tensor.shape[-2] == adj_tensor.shape[-2] == adj_tensor.shape[-1], 'Node dimension should fit'

        # Preprocess adj matrix
        if self.preprocess_adj:
            with torch.set_grad_enabled(self.node_self_importance.requires_grad):
                adj_tensor = self.preprocess_adj_to_adj_hat(adj_tensor, self.node_self_importance)

        # forward pass
        output_tensor = feat_tensor
        for layer in self.net:
            if isinstance(layer, GraphConvLayer):
                output_tensor = layer(output_tensor, adj_tensor)
            else:
                output_tensor = layer(output_tensor)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0], 'Out num_dims should fit'
        assert output_tensor.shape[-1] == self.output_features, 'Output feature dim should fit'

        return {self.out_keys[0]: output_tensor}

    def build_layer_dict(self) -> OrderedDict:
        """Compiles a block-specific dictionary of network layers.

        This could be overwritten by derived layers (e.g. to get a 'BatchNormalizedConvolutionBlock').

        :return: Ordered dictionary of torch modules [str, nn.Module].
        """
        layer_dict = OrderedDict()

        # treat first layer
        layer_dict["gcn_00"] = GraphConvLayer(in_features=self.input_features,
                                              out_features=self.hidden_features[0],
                                              bias=self.bias[0])
        layer_dict[f"{self.non_lins[0].__name__}_00"] = self.non_lins[0]()

        # treat remaining layers
        for i, h in enumerate(self.hidden_features[1:], start=1):
            layer_dict[f"conv_{i}0"] = GraphConvLayer(in_features=self.hidden_features[i - 1],
                                                      out_features=self.hidden_features[i],
                                                      bias=self.bias[i])
            layer_dict[f"{self.non_lins[i].__name__}_{i}0"] = self.non_lins[i]()
        return layer_dict

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f'({self.non_lins[0].__name__})' if len(set(self.non_lins)) == 1 else \
            f'({[non_lin.__name__ for non_lin in self.non_lins]})'
        txt += "\n\t" + f"({self.input_features}->" + "->".join([f"{h}" for h in self.hidden_features]) + ")"
        txt += f'\n\tBias: {self.bias if len(set(self.bias)) > 1 else self.bias[0]}'
        if self.node_self_importance.requires_grad:
            txt += f'\n\tNode-self-importance-as-param: True'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
