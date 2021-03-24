"""Self attention graph conv block based on https://arxiv.org/pdf/1710.10903.pdf and
 https://github.com/meliketoy/graph-cnn.pytorch"""
from collections import OrderedDict
from typing import Union, List, Sequence, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.feed_forward.graph_conv import GraphAdjacencyMethods
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class GraphAttentionLayer(nn.Module):
    """Simple graph attention layer.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param alpha: Specify the negative slope of the leakyReLU.
    :param dropout: Specify the dropout to be applied.
    """

    def __init__(self, in_features: int, out_features: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.alpha = alpha
        self.dropout = dropout

        self.weight = nn.Parameter(torch.empty(size=(in_features, out_features), dtype=torch.float32))
        self.weight_a1 = nn.Parameter(torch.empty(size=(out_features, 1), dtype=torch.float32))
        self.weight_a2 = nn.Parameter(torch.empty(size=(out_features, 1), dtype=torch.float32))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    @override(nn.Module)
    def forward(self, xx: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the layer with the modified adjacency matrix a_hat.

        :param xx: Input feature tensor.
        :param adj_hat: Preprocessed (normalized in row and column) adjacency matrix.
        :return: Output tensor.
        """
        h = torch.matmul(xx, self.weight)

        f_1 = torch.matmul(h, self.weight_a1)
        f_2 = torch.matmul(h, self.weight_a2)
        e = self.leakyrelu(f_1 + f_2.transpose(-2, -1))

        zero_vec = -np.inf * torch.ones_like(e)
        attention = torch.where(adj_hat > 0, e, zero_vec)

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        return h_prime

    @override(nn.Module)
    def __repr__(self):
        txt = self.__class__.__name__ + ': (' \
              + str(self.in_features) + ' -> ' \
              + str(self.out_features) + f'), alpha: {self.alpha}, dropout: {self.dropout}'
        return txt


class GraphMultiHeadAttentionLayer(nn.Module):
    """Multi-head-graph-attention layer. This layer consists of equivalent graph attention layers which are computed,
    concatenated and then send through the activation :param non_lin

    :param n_heads: The number of attention heads the layer should have.
    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param alpha: Specify the negative slope of the leakyReLU.
    :param dropout: Specify the dropout to be applied.
    :param avg_out: Specify whether to average the output or if set to false to concatenate it.s
    """

    def __init__(self, n_heads: int, in_features: int, out_features: int, alpha: float, dropout: float, avg_out: bool):
        super().__init__()
        assert n_heads > 0
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = out_features
        self.avg_out = avg_out
        self.attentions = nn.ModuleList()
        for _ in range(self.n_heads):
            self.attentions.append(GraphAttentionLayer(in_features, out_features, alpha, dropout))

    @override(nn.Module)
    def forward(self, xx: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the layer with the modified adjacency matrix a_hat.

        :param xx: Input feature tensor.
        :param adj_hat: Preprocessed (normalized in row and column) adjacency matrix.
        :return: Output tensor.
        """
        attention_outs = []
        for idx, att in enumerate(self.attentions):
            attention_outs.append(att(xx, adj_hat))

        if self.avg_out:
            output_tensor = sum(attention_outs) / len(attention_outs)
        else:
            output_tensor = torch.cat(attention_outs, dim=-1)

        return output_tensor

    @override(nn.Module)
    def __repr__(self):
        """Return a string representation of the layer."""
        txt = self.__class__.__name__ + f': {self.n_heads} x [{str(self.attentions[0])}]'
        return txt


class GraphAttentionBlock(ShapeNormalizationBlock, GraphAdjacencyMethods):
    """A block containing multiple subsequent graph (multi-head) attention stacks.

    One convolution stack consists of one graph multi-head attention in addition to an activation layer.
    The block expects the input tensors to have the form:

    - Feature matrix: first in_key: (batch-dim, num-of-nodes, feature-dim)
    - Adjacency matrix: second in_key: (batch-dim, num-of-nodes, num-of-nodes) (also symmetric)

    And returns a tensor of the form (batch-dim, num-of-nodes, feature-out-dim).

    :param in_keys: Two keys identifying the feature matrix and adjacency matrix respectively.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_features: List containing the number of hidden features for hidden layers.
    :param non_lins: The non-linearity/ies to apply after each layer (the same in all layers, or a list corresponding
        to each layer).
    :param n_heads: The number of heads each stack should have. (default suggestion 8)
    :param attention_alpha: Specify the negative slope of the leakyReLU in each of the attention layers.
        parameter with init value :param node_self_importance. (default suggestion 0.2)
    :param avg_last_head_attentions: Specify whether to average the outputs from the attention head in the last layer
        of the attention stack. (default suggestion True or n_heads=0 in the last layer)
    :param attention_dropout: Specify the dropout to be within the layers applied on the computed attention.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], hidden_features: List[int],
                 non_lins: Union[str, type(nn.Module), List[str], List[type(nn.Module)]],
                 n_heads: Union[int, List[int]], attention_alpha: Union[List[float], float],
                 avg_last_head_attentions: bool,
                 attention_dropout: Union[float, List[float]]):

        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=[3, 3],
                         out_num_dims=3)

        # Assertions
        assert len(self.in_keys) == 2, 'There should be two input keys, feature matrix + adjacency matrix'
        assert self.in_shapes[0][-2] == self.in_shapes[1][-1], 'The node dimension of the feature matrix should be ' \
                                                               'the same as the adjacency matrix\'s rows and ' \
                                                               f'columns {self.in_shapes}'
        assert self.in_shapes[1][-1] == self.in_shapes[1][-2], 'The adjacency matrix has to be a square matrix'
        self.avg_last_head_attentions = avg_last_head_attentions

        # Specify dummy dict creation function for adjacency matrix:
        self.dummy_dict_creators[1] = self._dummy_symmetric_adj_tensor_factory(self.in_shapes[1])

        # Init class objects
        self.input_features = self.in_shapes[0][-1]
        self.hidden_features = hidden_features

        # Create list of heads for each layer
        self.n_heads: List[int] = n_heads if isinstance(n_heads, list) else [n_heads] * len(self.hidden_features)

        # The output features of this block are equivalent to the specified last hidden features if
        #   :param avg_last_head_attention is set to true, otherwise the last output will be concatenated and as such is
        #   equivalent to the number of last hidden features times the last specified number of heads
        self.output_features = self.hidden_features[-1] if self.avg_last_head_attentions else \
            self.hidden_features[-1] * self.n_heads[-1]

        # Create list of non-linearity's for each layer
        non_lins = non_lins if isinstance(non_lins, list) else [non_lins] * len(self.hidden_features)
        self.non_lins: List[type(nn.Module)] = [Factory(base_type=nn.Module).type_from_name(non_lin)
                                                for non_lin in non_lins]

        # Create list of dropout for each layer
        self.attention_dropout = attention_dropout if isinstance(attention_dropout, list) \
            else [attention_dropout] * len(self.hidden_features)

        # Create list of alpha for each layer
        self.attention_alpha = attention_alpha if isinstance(attention_alpha, list) \
            else [attention_alpha] * len(self.hidden_features)

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

        adj_bar_tensor = self.preprocess_adj_to_adj_bar(adj_tensor)

        # forward pass
        output_tensor = feat_tensor
        for idx, layer in enumerate(self.net):
            if isinstance(layer, GraphMultiHeadAttentionLayer):
                output_tensor = layer(output_tensor, adj_bar_tensor)
            else:
                output_tensor = layer(output_tensor)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0], 'Out num_dims should fit'
        assert output_tensor.shape[-1] == self.output_features, 'Output feature dim should fit ' \
                                                                f'[{output_tensor.shape[-1]} vs {self.output_features})'

        return {self.out_keys[0]: output_tensor}

    def build_layer_dict(self) -> OrderedDict:
        """Compiles a block-specific dictionary of network layers.

        This could be overwritten by derived layers (e.g. to get a 'BatchNormalizedConvolutionBlock').

        :return: Ordered dictionary of torch modules [str, nn.Module].
        """
        layer_dict = OrderedDict()
        # treat first layer
        layer_dict[f"gat_00"] = GraphMultiHeadAttentionLayer(
            in_features=self.input_features, out_features=self.hidden_features[0], alpha=self.attention_alpha[0],
            n_heads=self.n_heads[0], dropout=self.attention_dropout[0],
            avg_out=self.avg_last_head_attentions and len(self.hidden_features) == 1
        )
        layer_dict[f"{self.non_lins[0].__name__}_00"] = self.non_lins[0]()

        # treat remaining layers
        for i, h in enumerate(self.hidden_features[1:], start=1):
            # Mulit-head attention
            layer_dict[f"gat_{i}0"] = GraphMultiHeadAttentionLayer(
                in_features=self.hidden_features[i - 1] * self.n_heads[i - 1], out_features=self.hidden_features[i],
                alpha=self.attention_alpha[i], n_heads=self.n_heads[i], dropout=self.attention_dropout[i],
                avg_out=self.avg_last_head_attentions and i == len(self.hidden_features) - 1
            )
            # Non-lin
            layer_dict[f"{self.non_lins[i].__name__}_{i}0"] = self.non_lins[i]()

        return layer_dict

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f'({self.non_lins[0].__name__})' if len(set(self.non_lins)) == 1 else \
            f'({[non_lin.__name__ for non_lin in self.non_lins]})'
        txt += "\n\t" + f"({self.input_features}->" + "->".join([f"{h} x {self.n_heads[idx]}"
                                                                 for idx, h in enumerate(self.hidden_features)]) \
               + f'->{self.output_features})'
        txt += f'\n\talpha: ({self.attention_alpha})' if len(
            set(self.attention_alpha)) == 1 else f'({[aa for aa in self.attention_alpha]})'
        txt += f'\n\tavg last head attentions: ' + str(self.avg_last_head_attentions)
        txt += f'\n\tdropout:' + (
            f'{self.attention_dropout}' if len(set(self.attention_dropout)) > 1 else f'{self.attention_dropout[0]}')
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
